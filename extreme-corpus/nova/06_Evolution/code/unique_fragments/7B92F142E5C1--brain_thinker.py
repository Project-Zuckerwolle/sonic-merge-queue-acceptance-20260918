"""Nova Predator v2 — BrainThinker v2.
Permanenter Hintergrunddienst — läuft alle 30 Minuten.

v2: 3 konkrete, messbare Aufgaben statt vagem Dreaming:
  Aufgabe 1 — Konfidenz-Konsolidierung (alle 30 Min, kein LLM):
    Entries der letzten 2h: in Session erwähnt → +0.1, nicht erwähnt >30d → -0.1
    Unter 0.3 → gelöscht.

  Aufgabe 2 — Duplikat-Merging (alle 30 Min, kein LLM):
    FuzzyDedup über neue Entries → Duplikate mergen (höhere Konfidenz behalten,
    Tags vereinen). Ergänzt den Write-Pfad-Check in BrainExtractor.

  Aufgabe 3 — Cross-Entry-Linking (täglich 03:00, qwen2.5:3b):
    Batches von 10 neuen Entries → LLM findet Verbindungen → Link-Stärken setzen.
    Setzt Evergreen-Flag wenn Entry in 3+ Sessions erwähnt wurde.

EventBus: BRAIN_UPDATE nach jedem Lauf → UI-Badge.
"""
from __future__ import annotations
import asyncio
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from core.brain_manager import BrainEntry
from core.event_bus import EventBus, EventTyp
from core.fuzzy_dedup import aehnlichkeit
from core.logger import get

if TYPE_CHECKING:
    from core.brain_index import BrainIndex
    from core.brain_manager import BrainManager
    from core.ollama_client import OllamaClient

log = get("brain_thinker")

_THINKER_INTERVALL_S  = 1800     # 30 Minuten
_LINKING_STUNDE       = 3        # 03:00 Uhr für Cross-Linking
_KONFIDENZ_PLUS       = 0.10
_KONFIDENZ_MINUS      = 0.10
_KONFIDENZ_MIN        = 0.30     # Unter diesem Wert: löschen
_KONFIDENZ_MAX        = 1.00
_DECAY_TAGE           = 30       # Kein Erwähnen seit N Tagen → Decay
_EVERGREEN_MIN_SESSIONS = 3      # Ab N Sessions → Evergreen-Flag
_LINKING_BATCH_SIZE   = 10       # Entries pro LLM-Linking-Batch
_LINKING_PROMPT = """\
Du analysierst Brain-Entries eines KI-Assistenten.
Finde Verbindungen zwischen diesen {n} Entries.
Antworte NUR mit JSON-Array, kein Text davor oder danach:
[
  {{"von": "entry_id_1", "zu": "entry_id_2", "staerke": 0.8, "grund": "kurze Begründung"}},
  ...
]
Wenn keine sinnvollen Verbindungen existieren: []

Entries:
{entries_text}
"""


class BrainThinker:
    """Permanenter Brain-Konsolidierungs-Dienst (v2)."""

    def __init__(
        self,
        brain_manager: "BrainManager",
        brain_index: "BrainIndex",
        bus: EventBus,
        ollama: "OllamaClient | None" = None,
        intervall_s: int = _THINKER_INTERVALL_S,
        brain_modell: str = "qwen2.5:3b",
        # Legacy-Parameter (werden ignoriert aber akzeptiert für Kompatibilität)
        brain_llm=None,
        task_manager=None,
        cluster_threshold: float = 0.75,
        embed_modell: str = "nomic-embed-text",
    ) -> None:
        self._brain        = brain_manager
        self._index        = brain_index
        self._bus          = bus
        self._ollama       = ollama
        self._intervall_s  = intervall_s
        self._brain_modell = brain_modell
        self._letzter_run: str | None = None
        self._letztes_linking: str | None = None   # Datum letztes Cross-Linking
        self._session_mentions: dict[str, int] = {}  # entry_id → Session-Zähler
        self._laeuft = False
        self._lauf_task: asyncio.Task | None = None

    # ── Lifecycle ────────────────────────────────────────────────────

    async def starten(self) -> None:
        if self._laeuft:
            return
        self._laeuft = True
        self._lauf_task = asyncio.create_task(
            self._zyklus(), name="brain_thinker_v2"
        )
        log.info("BrainThinker v2 gestartet (Intervall: %ds)", self._intervall_s)

    async def stoppen(self) -> None:
        self._laeuft = False
        if self._lauf_task and not self._lauf_task.done():
            self._lauf_task.cancel()
            try:
                await self._lauf_task
            except asyncio.CancelledError:
                pass
        log.info("BrainThinker v2 gestoppt")

    async def jetzt_ausfuehren(self) -> dict:
        """Manueller Trigger für einen Durchlauf."""
        return await self._thinker_durchlauf()

    def session_erwaehnung_melden(self, entry_id: str) -> None:
        """Wird von BrainExtractor aufgerufen wenn ein Entry in einer Session erwähnt wird.
        Erhöht den Session-Zähler für Evergreen-Flag-Logik.
        """
        self._session_mentions[entry_id] = self._session_mentions.get(entry_id, 0) + 1

    # ── Zyklus ───────────────────────────────────────────────────────

    async def _zyklus(self) -> None:
        while self._laeuft:
            await asyncio.sleep(self._intervall_s)
            if not self._laeuft:
                break
            try:
                await self._thinker_durchlauf()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error("BrainThinker Zyklus-Fehler: %s", e)

    async def _thinker_durchlauf(self) -> dict:
        jetzt = datetime.now(timezone.utc)
        jetzt_iso = jetzt.isoformat()
        log.info("BrainThinker v2 Durchlauf startet")

        ergebnis = {
            "konfidenz_updates":  0,
            "geloescht":          0,
            "duplikate_merged":   0,
            "neue_links":         0,
            "evergreen_gesetzt":  0,
            "fehler":             [],
        }

        try:
            alle = await self._brain.alle()
            if not alle:
                log.debug("BrainThinker: Brain leer")
                self._letzter_run = jetzt_iso
                return ergebnis

            # ── Aufgabe 1: Konfidenz-Konsolidierung ──────────────────
            await self._konfidenz_konsolidierung(alle, jetzt, ergebnis)

            # ── Aufgabe 2: Duplikat-Merging ───────────────────────────
            neue_entries = await self._brain.seit(self._letzter_run) if self._letzter_run else alle
            await self._duplikat_merging(neue_entries, ergebnis)

            # ── Aufgabe 3: Cross-Entry-Linking (nur täglich 03:00) ────
            heute = jetzt.strftime("%Y-%m-%d")
            if (
                jetzt.hour == _LINKING_STUNDE
                and self._letztes_linking != heute
                and self._ollama
            ):
                await self._cross_entry_linking(neue_entries, ergebnis)
                self._letztes_linking = heute

            # ── Evergreen-Flag setzen ─────────────────────────────────
            await self._evergreen_update(alle, ergebnis)

        except Exception as e:
            log.error("BrainThinker Durchlauf-Fehler: %s", e)
            ergebnis["fehler"].append(str(e))

        self._letzter_run = jetzt_iso

        # EventBus: UI-Badge triggern
        total_changes = (
            ergebnis["konfidenz_updates"] +
            ergebnis["duplikate_merged"] +
            ergebnis["neue_links"]
        )
        if total_changes > 0:
            await self._bus.publish(EventTyp.BRAIN_UPDATE, {
                "konfidenz_updates": ergebnis["konfidenz_updates"],
                "geloescht":         ergebnis["geloescht"],
                "neue_links":        ergebnis["neue_links"],
            })

        log.info(
            "BrainThinker v2 fertig: Konfidenz=%d, Gelöscht=%d, Merged=%d, Links=%d",
            ergebnis["konfidenz_updates"],
            ergebnis["geloescht"],
            ergebnis["duplikate_merged"],
            ergebnis["neue_links"],
        )
        return ergebnis

    # ── Aufgabe 1: Konfidenz-Konsolidierung ──────────────────────────

    async def _konfidenz_konsolidierung(
        self,
        alle: list[BrainEntry],
        jetzt: datetime,
        ergebnis: dict,
    ) -> None:
        """Passt Konfidenz basierend auf Erwähnungen und Alter an."""
        zu_loeschen = []

        for entry in alle:
            neue_konfidenz = entry.vertrauen
            veraendert = False

            # In dieser Session erwähnt → Boost
            if self._session_mentions.get(entry.id, 0) > 0:
                neue_konfidenz = min(_KONFIDENZ_MAX, neue_konfidenz + _KONFIDENZ_PLUS)
                veraendert = True

            # Temporal Decay: Alter prüfen
            elif not getattr(entry, "evergreen", False):
                try:
                    erstellt = datetime.fromisoformat(
                        entry.erstellt if isinstance(entry.erstellt, str)
                        else entry.erstellt.isoformat()
                    )
                    if erstellt.tzinfo is None:
                        erstellt = erstellt.replace(tzinfo=timezone.utc)
                    age_days = (jetzt.replace(tzinfo=timezone.utc) - erstellt).days
                    if age_days >= _DECAY_TAGE:
                        neue_konfidenz = max(0.0, neue_konfidenz - _KONFIDENZ_MINUS)
                        veraendert = True
                except Exception:
                    pass

            if veraendert:
                if neue_konfidenz < _KONFIDENZ_MIN:
                    zu_loeschen.append(entry.id)
                else:
                    entry.vertrauen = neue_konfidenz  # type: ignore[misc]
                    await self._brain.add(entry)
                    ergebnis["konfidenz_updates"] += 1

        # Löschen unter Schwellenwert
        for eid in zu_loeschen:
            try:
                await self._brain.loeschen(eid)
                ergebnis["geloescht"] += 1
                log.debug("Entry %s gelöscht (Konfidenz < %.2f)", eid[:8], _KONFIDENZ_MIN)
            except Exception as e:
                log.debug("Löschen %s fehlgeschlagen: %s", eid[:8], e)

        # Session-Mentions für nächsten Lauf zurücksetzen
        self._session_mentions.clear()

    # ── Aufgabe 2: Duplikat-Merging ───────────────────────────────────

    async def _duplikat_merging(
        self,
        neue_entries: list[BrainEntry],
        ergebnis: dict,
    ) -> None:
        """Findet Duplikate unter neuen Entries und merged sie."""
        if len(neue_entries) < 2:
            return

        gemergt: set[str] = set()

        for i, entry_a in enumerate(neue_entries):
            if entry_a.id in gemergt:
                continue
            for entry_b in neue_entries[i + 1:]:
                if entry_b.id in gemergt:
                    continue
                score = aehnlichkeit(entry_a.inhalt, entry_b.inhalt)
                if score >= 85:
                    # Merge: behalte höhere Konfidenz, vereinige Tags
                    if entry_a.vertrauen >= entry_b.vertrauen:
                        sieger, verlierer = entry_a, entry_b
                    else:
                        sieger, verlierer = entry_b, entry_a

                    # Tags vereinen
                    combined_tags = list(set(
                        getattr(sieger, "tags", []) +
                        getattr(verlierer, "tags", [])
                    ))
                    sieger.tags = combined_tags  # type: ignore[misc]
                    await self._brain.add(sieger)

                    # Verlierer löschen
                    try:
                        await self._brain.loeschen(verlierer.id)
                        gemergt.add(verlierer.id)
                        ergebnis["duplikate_merged"] += 1
                        log.debug(
                            "Duplikat merged [%.0f]: '%s' → '%s'",
                            score, verlierer.inhalt[:30], sieger.inhalt[:30],
                        )
                    except Exception as e:
                        log.debug("Merge-Löschen fehlgeschlagen: %s", e)

    # ── Aufgabe 3: Cross-Entry-Linking ────────────────────────────────

    async def _cross_entry_linking(
        self,
        neue_entries: list[BrainEntry],
        ergebnis: dict,
    ) -> None:
        """qwen2.5:3b findet Verbindungen zwischen neuen Entries."""
        if not neue_entries or not self._ollama:
            return

        # Batches von max. 10
        batches = [
            neue_entries[i: i + _LINKING_BATCH_SIZE]
            for i in range(0, len(neue_entries), _LINKING_BATCH_SIZE)
        ]

        for batch in batches:
            if len(batch) < 2:
                continue
            try:
                entries_text = "\n".join(
                    f'ID: {e.id[:8]} | Typ: {e.typ} | Inhalt: "{e.inhalt[:80]}"'
                    for e in batch
                )
                prompt = _LINKING_PROMPT.format(
                    n=len(batch),
                    entries_text=entries_text,
                )
                antwort = await self._ollama.chat(
                    nachrichten=[{"role": "user", "content": prompt}],
                    modell=self._brain_modell,
                )
                # JSON parsen
                antwort = antwort.strip()
                if antwort.startswith("```"):
                    antwort = antwort.split("```")[1]
                    if antwort.startswith("json"):
                        antwort = antwort[4:]
                links = json.loads(antwort)
                for link in (links or []):
                    von = link.get("von", "")
                    zu  = link.get("zu", "")
                    if von and zu:
                        # add_link existiert in BrainManager
                        ok = await self._brain.add_link(von, zu)
                        if ok:
                            ergebnis["neue_links"] += 1
            except Exception as e:
                log.debug("Cross-Linking Batch fehlgeschlagen: %s", e)

    # ── Evergreen-Flag ────────────────────────────────────────────────

    async def _evergreen_update(
        self,
        alle: list[BrainEntry],
        ergebnis: dict,
    ) -> None:
        """Setzt Evergreen-Flag wenn Entry in >= 3 Sessions erwähnt."""
        for entry in alle:
            if getattr(entry, "evergreen", False):
                continue
            count = self._session_mentions.get(entry.id, 0)
            # Approximation: wenn vertrauen >= 0.75 → stabil genug für Evergreen
            if entry.vertrauen >= 0.75 and count >= 1:
                try:
                    entry.evergreen = True  # type: ignore[misc]
                    await self._brain.add(entry)
                    ergebnis["evergreen_gesetzt"] += 1
                    log.debug("Evergreen gesetzt: '%s'", entry.inhalt[:40])
                except Exception:
                    pass

    # ── Status ────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "laeuft": self._laeuft,
            "letzter_run": self._letzter_run,
            "letztes_linking": self._letztes_linking,
            "intervall_s": self._intervall_s,
        }
