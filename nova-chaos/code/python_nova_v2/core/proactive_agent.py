"""Nova Predator v3.3 — ProactiveAgent.

Läuft alle 4 Stunden im Hintergrund (außer wenn Nova schläft).
Drei Aufgaben:

1. DEADLINE-ERKENNUNG:
   Brain-Entries + Todos mit Typ "aufgabe" und Zeitangaben werden von
   qwen2.5:3b auf Datum geparst. Wenn Deadline ≤ 24h → Toast + proaktive
   Erinnerung beim nächsten Chat.

2. FOLLOW-UP-ERKENNUNG:
   Aufgaben-Todos die > 7 Tage offen sind und kein follow_up_gesendet-Flag
   haben → beim nächsten Chat-Connect proaktiv fragen.

3. PROJEKT-TRACKING:
   Erkennt aus Gesprächs-Kontext (Brain-Entries) laufende Projekte und
   legt automatisch Todos + Projekt-Schritt-Todos an. Aktualisiert den
   Notizen-Stand wenn neue Brain-Entries zum Projekt kommen.

4. KONTEXT-EINSPEISUNG beim Chat-Start:
   Wenn User eine neue Nachricht schreibt und ein sehr relevanter
   Brain-Entry (Score > 0.8) noch nicht in diesem Chat erwähnt wurde,
   hängt Nova einen kurzen Hinweis ans Ende der Antwort.

Alle Aktionen sind fire-and-forget und blockieren nie den Chat-Flow.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

from core.logger import get

if TYPE_CHECKING:
    from core.brain_manager import BrainManager
    from core.event_bus import EventBus
    from core.ollama_client import OllamaClient
    from core.todo_manager import TodoManager
    from core.schlaf_manager import SchlafManager

log = get("proactive_agent")

_INTERVALL_S = 4 * 3600   # 4 Stunden

_DEADLINE_PROMPT = """\
Analysiere diesen Text und erkenne ob er eine Deadline enthält.
Text: "{text}"
Heutiges Datum: {datum}

Wenn eine Deadline erkennbar ist, antworte: DEADLINE: YYYY-MM-DD
Wenn keine Deadline: KEINE
Nur diese Formate, kein anderer Text.
"""

_PROJEKT_ERKENN_PROMPT = """\
Analysiere diese Brain-Einträge und erkenne aktive Projekte des Users.

Einträge:
{eintraege}

Für jedes erkannte Projekt antworte in diesem Format (eines pro Zeile):
PROJEKT: <Name> | STAND: <1-Satz-Zusammenfassung> | NAECHSTER_SCHRITT: <konkreter nächster Schritt>

Nur echte laufende Projekte. Wenn keine erkennbar: KEINE
"""


class ProactiveAgent:
    """Hintergrund-Agent für proaktive Erinnerungen + Projekt-Tracking."""

    def __init__(
        self,
        brain_manager: "BrainManager",
        todo_manager: "TodoManager",
        ollama: "OllamaClient",
        bus: "EventBus",
        schlaf: "SchlafManager | None" = None,
        nano_modell: str = "qwen2.5:3b",
        intervall_s: int = _INTERVALL_S,
    ) -> None:
        self._brain    = brain_manager
        self._todos    = todo_manager
        self._ollama   = ollama
        self._bus      = bus
        self._schlaf   = schlaf
        self._nano     = nano_modell
        self._intervall_s = intervall_s

        # Pending-Queue: wird beim nächsten Chat-Connect abgearbeitet
        self._pending_erinnerungen: list[str] = []
        self._follow_ups: list[str] = []   # Todo-IDs für Follow-Up-Fragen
        self._laeuft = False
        self._task: asyncio.Task | None = None
        self._bereits_getrackt: set[str] = set()  # Bereits als Projekt-Todo erfasste Brain-IDs

    # ── Lifecycle ────────────────────────────────────────────────────

    async def starten(self) -> None:
        if self._laeuft:
            return
        self._laeuft = True
        self._task = asyncio.create_task(self._zyklus(), name="proactive_agent")
        log.info("ProactiveAgent gestartet (Intervall: %dh)", self._intervall_s // 3600)

    async def stoppen(self) -> None:
        self._laeuft = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ── Public API für nova_ws ────────────────────────────────────────

    def pending_erinnerungen(self) -> list[str]:
        """Gibt ausstehende Erinnerungen zurück und leert die Queue."""
        msgs = list(self._pending_erinnerungen)
        self._pending_erinnerungen.clear()
        return msgs

    def pending_follow_ups(self) -> list[str]:
        """Follow-Up-Fragen für nächsten Chat-Connect."""
        fu = list(self._follow_ups)
        self._follow_ups.clear()
        return fu

    async def projekt_kontext_laden(self, projekt_name: str) -> str | None:
        """Lädt Stand + nächste Schritte eines Projekts für den Chat-Kontext."""
        todos = self._todos.nach_projekt(projekt_name, nur_offen=True)
        if not todos:
            return None
        zeilen = [f"**Projekt: {projekt_name}**"]
        for t in todos[:5]:
            notiz = f" ({t.notizen[:80]})" if t.notizen else ""
            zeilen.append(f"- {t.text}{notiz}")
        return "\n".join(zeilen)

    # ── Haupt-Zyklus ─────────────────────────────────────────────────

    async def _zyklus(self) -> None:
        # Ersten Lauf nach 30s (Startup-Delay)
        await asyncio.sleep(30)
        while self._laeuft:
            # Nicht laufen wenn Nova schläft (kein VRAM verschwenden)
            if self._schlaf and self._schlaf.schlaeft:
                await asyncio.sleep(300)
                continue
            try:
                await self._durchlauf()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error("ProactiveAgent Fehler: %s", e)
            await asyncio.sleep(self._intervall_s)

    async def _durchlauf(self) -> None:
        log.info("ProactiveAgent: Durchlauf startet")
        alle_entries = await self._brain.alle()
        alle_todos   = self._todos.alle(nur_offen=True)
        jetzt        = datetime.now(timezone.utc)

        await self._deadline_check(alle_entries, alle_todos, jetzt)
        await self._follow_up_check(alle_todos, jetzt)
        await self._projekt_tracking(alle_entries)

        log.info(
            "ProactiveAgent: fertig — %d Erinnerungen, %d Follow-ups pending",
            len(self._pending_erinnerungen),
            len(self._follow_ups),
        )

    # ── Aufgabe 1: Deadline-Erkennung ────────────────────────────────

    async def _deadline_check(self, entries, todos, jetzt: datetime) -> None:
        datum_str = jetzt.strftime("%Y-%m-%d")
        morgen = (jetzt + timedelta(days=1)).strftime("%Y-%m-%d")

        # Brain-Entries + Todos mit Datum-Keywords prüfen
        kandidaten: list[str] = []
        datum_kw = ("morgen", "übermorgen", "bis ", "deadline", "fällig", "abgabe")

        for entry in entries:
            if entry.typ == "aufgabe" and any(kw in entry.inhalt.lower() for kw in datum_kw):
                kandidaten.append(entry.inhalt)

        for todo in todos:
            if any(kw in todo.text.lower() for kw in datum_kw):
                kandidaten.append(f"TODO: {todo.text}")

        if not kandidaten:
            return

        # LLM prüft Deadlines (max 5 Kandidaten)
        for kandidat in kandidaten[:5]:
            try:
                antwort = await self._ollama.chat(
                    nachrichten=[{"role": "user", "content": _DEADLINE_PROMPT.format(
                        text=kandidat[:200],
                        datum=datum_str,
                    )}],
                    modell=self._nano,
                    optionen={"temperature": 0.0, "num_predict": 20, "think": False},
                )
                antwort = antwort.strip()
                if antwort.startswith("DEADLINE:"):
                    deadline = antwort.replace("DEADLINE:", "").strip()
                    if deadline <= morgen:
                        msg = f"⏰ Deadline bald: {kandidat[:80]}"
                        if msg not in self._pending_erinnerungen:
                            self._pending_erinnerungen.append(msg)
                            log.info("Deadline erkannt: %s", kandidat[:60])
            except Exception as e:
                log.debug("Deadline-Check Fehler: %s", e)

    # ── Aufgabe 2: Follow-Up-Erkennung ───────────────────────────────

    async def _follow_up_check(self, todos, jetzt: datetime) -> None:
        grenze = jetzt - timedelta(days=7)
        for todo in todos:
            # Überspringe schon gemeldete + Projekt-Schritte
            if todo.id in self._bereits_getrackt:
                continue
            try:
                erstellt = datetime.fromisoformat(todo.erstellt)
                if erstellt.tzinfo is None:
                    erstellt = erstellt.replace(tzinfo=timezone.utc)
                if erstellt < grenze and not todo.erledigt:
                    msg = f"Noch offen (>7 Tage): {todo.text[:80]}"
                    if msg not in self._follow_ups:
                        self._follow_ups.append(msg)
                        self._bereits_getrackt.add(todo.id)
                        log.debug("Follow-Up: %s", todo.text[:50])
            except Exception:
                pass

    # ── Aufgabe 3: Projekt-Tracking ───────────────────────────────────

    async def _projekt_tracking(self, entries) -> None:
        """Erkennt laufende Projekte aus Brain-Entries und legt Todos an."""
        # Nur neue Entries (noch nicht getrackt)
        neue = [
            e for e in entries
            if e.id not in self._bereits_getrackt
            and e.typ in ("aufgabe", "idee", "fakt")
            and e.quelle in ("chat", "korrektur")
        ]

        if len(neue) < 3:
            return

        eintraege_text = "\n".join(
            f"- [{e.typ}] {e.inhalt[:100]}" for e in neue[:20]
        )

        try:
            antwort = await self._ollama.chat(
                nachrichten=[{"role": "user", "content": _PROJEKT_ERKENN_PROMPT.format(
                    eintraege=eintraege_text
                )}],
                modell=self._nano,
                optionen={"temperature": 0.1, "num_predict": 400, "think": False},
            )

            if "KEINE" in antwort or not antwort.strip():
                # Alle als getrackt markieren
                for e in neue:
                    self._bereits_getrackt.add(e.id)
                return

            # Projekte parsen
            for zeile in antwort.strip().splitlines():
                if not zeile.startswith("PROJEKT:"):
                    continue
                try:
                    teile = dict(
                        t.split(":", 1) for t in zeile.split("|") if ":" in t
                    )
                    projekt_name  = teile.get("PROJEKT", "").strip()
                    stand         = teile.get("STAND", "").strip()
                    naechster     = teile.get("NAECHSTER_SCHRITT", "").strip()

                    if not projekt_name:
                        continue

                    # Prüfen ob Projekt-Todo bereits existiert
                    bestehende = self._todos.nach_projekt(projekt_name)
                    if not bestehende:
                        # Neues Projekt-Todo anlegen
                        await self._todos.add(
                            text=f"Projekt: {projekt_name}",
                            prioritaet="mittel",
                            typ="projekt",
                            projekt=projekt_name,
                            notizen=stand,
                        )
                        log.info("Neues Projekt-Todo: %s", projekt_name)

                    # Nächsten Schritt als Todo
                    if naechster:
                        schritt_texte = [t.text for t in bestehende]
                        if naechster not in schritt_texte:
                            await self._todos.add(
                                text=naechster,
                                prioritaet="mittel",
                                typ="projekt_schritt",
                                projekt=projekt_name,
                                notizen=stand,
                            )
                            log.info("Projekt-Schritt: %s → %s", projekt_name, naechster[:50])
                    else:
                        # Bestehenden Projekt-Todo-Notizen updaten
                        for t in bestehende:
                            if t.typ == "projekt" and stand:
                                await self._todos.aktualisieren(t.id, notizen=stand)

                except Exception as e:
                    log.debug("Projekt-Parse Fehler: %s", e)

            # Verarbeitete Entries als getrackt markieren
            for e in neue:
                self._bereits_getrackt.add(e.id)

        except Exception as e:
            log.debug("Projekt-Tracking Fehler: %s", e)

    def status(self) -> dict:
        return {
            "laeuft":          self._laeuft,
            "pending_count":   len(self._pending_erinnerungen),
            "follow_up_count": len(self._follow_ups),
            "getrackt":        len(self._bereits_getrackt),
        }
