"""Nova Predator v1 — Briefing.
Erzeugt tägliche Briefings aus Brain-Entries, Todos und Session-Facts.
Speichert als Markdown in data/briefings/.
Python 3.14: aiofiles, asyncio.to_thread(), keine deprecated APIs.
"""
from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles

from core.event_bus import EventBus, EventTyp
from core.logger import get

if TYPE_CHECKING:
    from core.brain_manager import BrainManager
    from core.ollama_client import OllamaClient
    from core.todo_manager import TodoManager

log = get("briefing")


class BriefingGenerator:
    def __init__(
        self,
        brain_manager: "BrainManager",
        todo_manager: "TodoManager",
        ollama: "OllamaClient",
        bus: EventBus,
        chat_modell: str = "gemma4:e4b",
        briefing_pfad: str | Path = "data/briefings",
    ) -> None:
        self._brain = brain_manager
        self._todos = todo_manager
        self._ollama = ollama
        self._bus = bus
        self._chat_modell = chat_modell
        self._briefing_pfad = Path(briefing_pfad)
        self._letztes_briefing: str | None = None

    async def generiere(self) -> str:
        """Generiert ein Daily Briefing. Gibt Markdown-Text zurück."""
        jetzt = datetime.now(timezone.utc)
        datum_str = jetzt.strftime("%Y-%m-%d")

        # Daten sammeln
        alle_entries = await self._brain.alle()
        offene_todos = self._todos.alle(nur_offen=True)
        projekte     = self._todos.projekte(nur_offen=True)
        durchbruch_entries = [e for e in alle_entries if "durchbruch" in e.tags or e.score >= 0.75]
        neueste_fakten = sorted(
            [e for e in alle_entries if e.typ in ("fakt", "praeferenz")],
            key=lambda e: e.erstellt,
            reverse=True,
        )[:5]

        # Briefing zusammenstellen
        abschnitte = [f"# Nova Briefing — {datum_str}\n"]

        # ── Aktive Projekte (v3.3) ──────────────────────────────────
        if projekte:
            abschnitte.append("## Aktive Projekte")
            for projekt_name, todos in sorted(projekte.items()):
                offen = [t for t in todos if not t.erledigt]
                erledigt_count = len([t for t in todos if t.erledigt])
                abschnitte.append(f"\n### {projekt_name}")
                abschnitte.append(f"*{len(offen)} offen, {erledigt_count} erledigt*")
                for t in offen[:5]:
                    notiz = f" — {t.notizen[:60]}" if t.notizen else ""
                    prio = f"[{t.prioritaet}] " if t.prioritaet == "hoch" else ""
                    abschnitte.append(f"- {prio}{t.text}{notiz}")
            abschnitte.append("")

        if durchbruch_entries:
            abschnitte.append("## Durchbruch-Ideen")
            for e in durchbruch_entries[:3]:
                abschnitte.append(f"- **[Score {e.score:.2f}]** {e.inhalt}")
            abschnitte.append("")

        # Nicht-Projekt Todos
        solo_todos = [t for t in offene_todos if not t.projekt]
        if solo_todos:
            abschnitte.append("## Offene Aufgaben")
            for t in solo_todos[:5]:
                prio = f"[{t.prioritaet}] " if t.prioritaet != "mittel" else ""
                abschnitte.append(f"- {prio}{t.text}")
            abschnitte.append("")

        if neueste_fakten:
            abschnitte.append("## Zuletzt gelernt")
            for e in neueste_fakten:
                abschnitte.append(f"- {e.inhalt}")
            abschnitte.append("")

        # LLM-Zusammenfassung (kurz, optional)
        if alle_entries and len(alle_entries) >= 3:
            try:
                zusammenfassung = await self._erstelle_zusammenfassung(
                    alle_entries[:10], projekte
                )
                if zusammenfassung:
                    abschnitte.append("## Nova denkt")
                    abschnitte.append(zusammenfassung)
                    abschnitte.append("")
            except Exception as e:
                log.debug("Briefing LLM-Zusammenfassung fehlgeschlagen: %s", e)

        abschnitte.append(f"*Generiert: {jetzt.strftime('%H:%M Uhr')}*")
        briefing_text = "\n".join(abschnitte)

        await self._speichern(datum_str, briefing_text)
        self._letztes_briefing = datum_str
        await self._bus.publish(EventTyp.SCHEDULED_TASK, {"typ": "briefing", "datum": datum_str})
        log.info("Briefing generiert: %s (%d Zeichen)", datum_str, len(briefing_text))
        return briefing_text

    async def heute_generiert(self) -> bool:
        """Prüft ob heute bereits ein Briefing existiert."""
        datum = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        pfad = self._briefing_pfad / f"briefing_{datum}.md"
        return await asyncio.to_thread(pfad.exists)

    async def lese_letztes(self) -> str | None:
        """Gibt das zuletzt generierte Briefing zurück."""
        self._briefing_pfad.mkdir(parents=True, exist_ok=True)
        dateien = sorted(self._briefing_pfad.glob("briefing_*.md"), reverse=True)
        if not dateien:
            return None
        async with aiofiles.open(dateien[0], encoding="utf-8") as f:
            return await f.read()

    async def _erstelle_zusammenfassung(self, entries, projekte: dict | None = None) -> str:
        """Lässt LLM eine kurze Zusammenfassung des aktuellen Wissensstands erstellen."""
        inhalte = "\n".join(f"- {e.inhalt[:100]}" for e in entries)
        projekt_info = ""
        if projekte:
            namen = ", ".join(list(projekte.keys())[:3])
            projekt_info = f"\nAktive Projekte: {namen}"
        prompt = (
            f"Fasse in 2-3 Sätzen zusammen was zuletzt gelernt wurde und was ansteht:{projekt_info}\n{inhalte}\n"
            f"Direkt und prägnant, kein 'Du hast...' oder 'Es wurde...'"
        )
        return await self._ollama.chat(
            nachrichten=[{"role": "user", "content": prompt}],
            modell=self._chat_modell,
            optionen={"temperature": 0.4, "num_predict": 150, "think": False},
        )

    async def _speichern(self, datum: str, text: str) -> None:
        self._briefing_pfad.mkdir(parents=True, exist_ok=True)
        pfad = self._briefing_pfad / f"briefing_{datum}.md"
        async with aiofiles.open(pfad, "w", encoding="utf-8") as f:
            await f.write(text)
