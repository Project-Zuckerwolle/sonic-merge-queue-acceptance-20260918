"""Nova Predator v1 Layer 6 — Memory-Bridge.

Trägt Ergebnisse aus Layer-6-Tasks in Brain und Session ein.
Kein starker Bruch zwischen Agent-Chat und Main-Chat:
  - Wichtige Fakten → Brain (persistent, für alle Chats sichtbar)
  - Task-Zusammenfassung → Session-Episodic (im nächsten Chat-Turn verfügbar)
  - Extrahierte Fakten → SessionFacts (sofort sichtbar im System-Prompt)

Regeln:
  - Niemals mehr als 5 Brain-Einträge pro Task (kein Spam)
  - Nur Fakten mit Substanz (> 20 Zeichen, kein Duplikat)
  - SessionFacts nur solange die aktuelle Session läuft
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from core.logger import get

if TYPE_CHECKING:
    from core.brain_manager import BrainManager, BrainEntry
    from layer6.apex_types import ApexTask
    from memory.session import Session

log = get("apex.memory_bridge")

_MAX_BRAIN_EINTRAEGE = 5
_MIN_FAKT_LAENGE     = 20


class MemoryBridge:
    """Verbindet Layer-6-Ergebnisse mit Brain und Session des Haupt-Chats."""

    def __init__(
        self,
        brain_manager: "BrainManager",
        session: "Session",
    ) -> None:
        self._brain   = brain_manager
        self._session = session

    async def nach_task(self, task: "ApexTask") -> dict:
        """Wird nach Abschluss eines Tasks aufgerufen.

        Trägt ein:
          1. Task-Zusammenfassung als Session-Turn (Episodic Memory)
          2. Extrahierte Fakten als Brain-Entries
          3. Key-Findings als SessionFacts

        Gibt Statistik zurück: {"brain": N, "session_facts": M}
        """
        stats = {"brain": 0, "session_facts": 0}

        if not task.ergebnis:
            return stats

        # ── 1. Task-Zusammenfassung in Session-History ───────────────
        # Erscheint im nächsten Main-Chat als Kontext
        zusammenfassung = (
            f"[Agent-Aufgabe abgeschlossen]\n"
            f"Aufgabe: {task.aufgabe}\n"
            f"Ergebnis: {task.ergebnis[:600]}\n"
            f"Verlauf: {task.zusammenfassung()}"
        )
        self._session.speichere_turn(
            f"[Agent] {task.aufgabe}",
            task.ergebnis[:800],
        )
        log.debug("Task-Zusammenfassung in Session eingetragen")

        # ── 2. Extrahierte Fakten → Brain ────────────────────────────
        # Nur Fakten mit Substanz, max. 5 pro Task
        valide_fakten = [
            f for f in task.brain_eintraege
            if len(f.strip()) >= _MIN_FAKT_LAENGE
        ][:_MAX_BRAIN_EINTRAEGE]

        for fakt_text in valide_fakten:
            await self._brain_eintrag_hinzufuegen(
                inhalt=fakt_text,
                quelle=f"apex_agent:{task.id[:8]}",
                tags=["apex", "agent", "automatisch"],
            )
            stats["brain"] += 1

        # Auch das Ergebnis selbst als Notiz im Brain
        if len(task.ergebnis) >= _MIN_FAKT_LAENGE:
            await self._brain_eintrag_hinzufuegen(
                inhalt=f"Agent-Ergebnis ({task.aufgabe[:60]}): {task.ergebnis[:400]}",
                quelle=f"apex_agent:{task.id[:8]}",
                tags=["apex", "ergebnis"],
                typ="notiz",
            )
            stats["brain"] += 1

        # ── 3. Key-Findings als SessionFacts (sofort im System-Prompt) ──
        # Nur 1-2 wichtigste Fakten als SessionFact
        for fakt in valide_fakten[:2]:
            self._session.fact_hinzufuegen(
                text=f"[Vom Agent recherchiert] {fakt}",
                konfidenz=0.85,
            )
            stats["session_facts"] += 1

        log.info(
            "Memory-Bridge: Task %s → %d Brain-Entries, %d SessionFacts",
            task.id[:8], stats["brain"], stats["session_facts"]
        )
        return stats

    async def fortschritts_eintrag(self, task_id: str, aufgabe: str,
                                    fortschritt: str) -> None:
        """Trägt Zwischen-Fortschritt als SessionFact ein (für lange Tasks).

        Wird bei Reflexions-Events aufgerufen damit der User im Main-Chat
        sehen kann was der Agent gerade macht.
        """
        self._session.fact_hinzufuegen(
            text=f"[Agent läuft] {aufgabe[:50]}: {fortschritt[:150]}",
            konfidenz=0.7,
        )

    async def _brain_eintrag_hinzufuegen(
        self,
        inhalt: str,
        quelle: str,
        tags: list[str] | None = None,
        typ: str = "fakt",
    ) -> None:
        """Fügt einen neuen BrainEntry hinzu — überspringt Duplikate."""
        from core.brain_manager import BrainEntry

        # Einfache Duplikat-Prüfung: Gleicher Inhalt (case-insensitive) vorhanden?
        inhalt_lower = inhalt.lower()
        alle = await self._brain.alle()
        for e in alle:
            if e.inhalt.lower() == inhalt_lower:
                log.debug("Brain-Duplikat übersprungen: %s", inhalt[:50])
                return

        entry = BrainEntry(
            id=str(uuid.uuid4()),
            typ=typ,
            inhalt=inhalt,
            quelle=quelle,
            tags=tags or ["apex"],
            vertrauen=0.8,
        )
        await self._brain.add(entry)
