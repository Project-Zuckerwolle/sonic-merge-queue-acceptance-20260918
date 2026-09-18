"""Nova Predator v2 — SubtaskQueue.
Persistente Queue für Subtasks mit Crash-Recovery-Fähigkeit.

Die Queue ist der Single-Source-of-Truth für den Fortschritt eines Builds:
  - Jede Änderung wird sofort auf Disk geschrieben
  - Beim nächsten Start kann die Queue geladen und weitergemacht werden
  - Thread-safe für asyncio (kein shared state außer über diese Klasse)

Gespeichert in: workspace_agent/subtask_queue.json
"""
from __future__ import annotations
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from core.logger import get
from models.orchestrator_state import OrchestratorState, SubtaskStateEntry, SubtaskStatus
from models.subtask_paket import SubtaskPaket
from models.ergebnis_paket import ErgebnisPaket

log = get("subtask_queue")

_DEFAULT_QUEUE_PFAD = Path("workspace_agent/subtask_queue.json")


class SubtaskQueue:
    """Thread-safe persistente Subtask-Queue.

    Verwendung:
        queue = SubtaskQueue()
        queue.initialisieren(projekt_id, plan_text, subtasks)
        await queue.speichern()

        nxt = queue.naechster()
        queue.markiere_running(nxt.paket['id'])
        ...
        queue.markiere_done(subtask_id, ergebnis)
        await queue.speichern()
    """

    def __init__(self, pfad: Path = _DEFAULT_QUEUE_PFAD) -> None:
        self._pfad  = pfad
        self._state: OrchestratorState | None = None
        self._lock  = asyncio.Lock()

    # ── Initialisierung ──────────────────────────────────────────────

    def initialisieren(
        self,
        projekt_id: str,
        plan_text: str,
        projektziel: str,
        subtasks: list[SubtaskPaket],
    ) -> None:
        """Initialisiert neue Queue aus Subtask-Liste."""
        self._state = OrchestratorState(
            projekt_id=projekt_id,
            plan_text=plan_text,
            projektziel=projektziel,
            gestartet_um=datetime.now(timezone.utc).isoformat(),
            letztes_update=datetime.now(timezone.utc).isoformat(),
            subtasks=[
                SubtaskStateEntry(paket=s.to_dict(), status="pending")
                for s in subtasks
            ],
        )
        log.info("Queue initialisiert: %d Subtasks", len(subtasks))

    async def laden(self) -> bool:
        """Lädt Queue von Disk (Crash-Recovery).

        Returns:
            True wenn geladen, False wenn keine Queue gefunden.
        """
        async with self._lock:
            state = await asyncio.to_thread(OrchestratorState.laden, str(self._pfad))
            if state:
                self._state = state
                f = state.fortschritt()
                log.info(
                    "Queue geladen: %d total, %d done, %d pending",
                    f["total"], f["done"], f["pending"],
                )
                return True
            return False

    async def speichern(self) -> None:
        """Speichert Queue auf Disk (async-safe)."""
        if not self._state:
            return
        self._state.letztes_update = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            await asyncio.to_thread(
                self._state.speichern, str(self._pfad)
            )

    async def loeschen(self) -> None:
        """Löscht Queue-Datei nach erfolgreichem Abschluss."""
        async with self._lock:
            def _del():
                p = self._pfad
                if p.exists():
                    p.unlink()
            await asyncio.to_thread(_del)
        log.info("Queue gelöscht: %s", self._pfad)

    # ── Queue-Operationen ────────────────────────────────────────────

    def naechster(self) -> SubtaskStateEntry | None:
        """Gibt nächsten Subtask zurück dessen Abhängigkeiten erfüllt sind."""
        if not self._state:
            return None
        return self._state.naechster_pending()

    def markiere_running(self, subtask_id: str) -> None:
        """Markiert Subtask als laufend."""
        entry = self._finde(subtask_id)
        if entry:
            entry.status = "running"
            entry.versuche += 1

    def markiere_done(
        self,
        subtask_id: str,
        ergebnis: ErgebnisPaket,
    ) -> None:
        """Markiert Subtask als abgeschlossen."""
        entry = self._finde(subtask_id)
        if entry:
            entry.status = "done"
            entry.ergebnis = ergebnis.to_dict()
            log.info("Subtask '%s' done ✓", subtask_id)

    def markiere_failed(
        self,
        subtask_id: str,
        feedback: str = "",
        ergebnis: ErgebnisPaket | None = None,
    ) -> None:
        """Markiert Subtask als fehlgeschlagen."""
        entry = self._finde(subtask_id)
        if entry:
            entry.status = "failed"
            entry.feedback = feedback
            if ergebnis:
                entry.ergebnis = ergebnis.to_dict()
            log.warning("Subtask '%s' failed: %s", subtask_id, feedback[:80])

    def markiere_retry(self, subtask_id: str, feedback: str) -> None:
        """Setzt Subtask zurück auf pending für Retry."""
        entry = self._finde(subtask_id)
        if entry:
            entry.status = "retrying"
            entry.feedback = feedback
            # Nach kurzer Pause: pending setzen damit naechster() ihn nimmt
            entry.status = "pending"

    # ── Status & Info ────────────────────────────────────────────────

    def fortschritt(self) -> dict:
        if not self._state:
            return {"total": 0, "done": 0, "failed": 0, "pending": 0, "prozent": 0}
        return self._state.fortschritt()

    def ist_abgeschlossen(self) -> bool:
        """True wenn keine pending Subtasks mehr."""
        f = self.fortschritt()
        return f["pending"] == 0

    def hat_fehler(self) -> bool:
        """True wenn mindestens ein Subtask failed ist."""
        return self.fortschritt()["failed"] > 0

    def alle_subtasks(self) -> list[SubtaskStateEntry]:
        if not self._state:
            return []
        return list(self._state.subtasks)

    def als_dict(self) -> dict:
        if not self._state:
            return {}
        return self._state.to_dict()

    @property
    def projekt_id(self) -> str:
        return self._state.projekt_id if self._state else ""

    @property
    def projektziel(self) -> str:
        return self._state.projektziel if self._state else ""

    # ── Intern ───────────────────────────────────────────────────────

    def _finde(self, subtask_id: str) -> SubtaskStateEntry | None:
        if not self._state:
            return None
        for entry in self._state.subtasks:
            if entry.paket.get("id") == subtask_id:
                return entry
        return None
