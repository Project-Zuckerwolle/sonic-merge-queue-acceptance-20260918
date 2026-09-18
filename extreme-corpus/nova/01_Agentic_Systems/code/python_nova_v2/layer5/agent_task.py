"""Nova Predator v1 Layer 5 — Agent-Task Datenstrukturen + Retry/Loop-Schutz."""
from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from core.logger import get

log = get("layer5.agent_task")


class AgentStatus(Enum):
    PENDING    = "pending"
    PLANNING   = "planning"
    RUNNING    = "running"
    VERIFYING  = "verifying"
    DONE       = "done"
    FAILED     = "failed"
    ABORTED    = "aborted"


@dataclass
class Subtask:
    id:          str
    beschreibung: str
    status:      AgentStatus = AgentStatus.PENDING
    ergebnis:    str = ""
    fehler:      str = ""
    versuche:    int = 0
    dateien:     list[str] = field(default_factory=list)
    started_at:  float = 0.0
    done_at:     float = 0.0

    @property
    def dauer_ms(self) -> int:
        if self.done_at and self.started_at:
            return int((self.done_at - self.started_at) * 1000)
        return 0


@dataclass
class AgentTask:
    id:          str
    aufgabe:     str
    projekt:     str
    workspace:   Path
    status:      AgentStatus    = AgentStatus.PENDING
    subtasks:    list[Subtask]  = field(default_factory=list)
    plan_text:   str            = ""
    rückfragen:  list[str]      = field(default_factory=list)
    log:         list[dict]     = field(default_factory=list)
    started_at:  float          = field(default_factory=time.time)
    done_at:     float          = 0.0

    def log_event(self, typ: str, **kwargs: Any) -> None:
        self.log.append({"typ": typ, "ts": time.time(), **kwargs})

    def aktueller_subtask(self) -> Subtask | None:
        for s in self.subtasks:
            if s.status == AgentStatus.RUNNING:
                return s
        return None

    def naechster_subtask(self) -> Subtask | None:
        for s in self.subtasks:
            if s.status == AgentStatus.PENDING:
                return s
        return None

    def alle_fertig(self) -> bool:
        return all(s.status in (AgentStatus.DONE, AgentStatus.FAILED)
                   for s in self.subtasks)

    def zusammenfassung(self) -> str:
        done   = sum(1 for s in self.subtasks if s.status == AgentStatus.DONE)
        failed = sum(1 for s in self.subtasks if s.status == AgentStatus.FAILED)
        total  = len(self.subtasks)
        return f"{done}/{total} Subtasks erfolgreich, {failed} fehlgeschlagen"


# ── Retry-Mechanik ───────────────────────────────────────────────────────────

class RetryMechanic:
    """Verwaltet Wiederholungsversuche für Subtasks.

    Nach MAX_RETRIES Versuchen wird der Subtask als fehlgeschlagen markiert.
    Jeder Retry bekommt ein Korrektur-Briefing von Gemma4.
    """

    MAX_RETRIES = 3

    def __init__(self) -> None:
        self._versuche: dict[str, int] = {}   # subtask_id → Anzahl Versuche

    def darf_retry(self, subtask: Subtask) -> bool:
        return self._versuche.get(subtask.id, 0) < self.MAX_RETRIES

    def registriere_versuch(self, subtask: Subtask) -> int:
        n = self._versuche.get(subtask.id, 0) + 1
        self._versuche[subtask.id] = n
        return n

    def korrektur_briefing(self, subtask: Subtask, fehler: str) -> str:
        n = self._versuche.get(subtask.id, 0)
        return (
            f"Versuch {n}/{self.MAX_RETRIES}. "
            f"Beim letzten Versuch ist folgender Fehler aufgetreten:\n{fehler}\n\n"
            f"Bitte korrigiere das Problem und versuche es erneut.\n"
            f"Aufgabe: {subtask.beschreibung}"
        )


# ── Loop-Detektor ────────────────────────────────────────────────────────────

class LoopDetector:
    """Erkennt Endlos-Schleifen im Agent-Loop.

    Drei Strategien gleichzeitig:
    1. Output-Hash: gleicher Output zweimal → Loop
    2. File-Progress: keine neue Datei in 3 Iterationen → Stagnation
    3. Subtask-Fingerprint: gleicher Subtask zweimal ohne Fortschritt
    """

    STAGNATION_LIMIT = 4   # Iterationen ohne neuen File-Write
    HASH_WINDOW      = 5   # Letzte N Output-Hashes prüfen

    def __init__(self, workspace: Path) -> None:
        self._ws             = workspace
        self._output_hashes: deque[str] = deque(maxlen=self.HASH_WINDOW)
        self._letzter_write  = 0     # Iteration des letzten File-Writes
        self._subtask_finger: deque[str] = deque(maxlen=4)
        self._iteration      = 0

    def naechste_iteration(self, output: str, subtask_id: str) -> None:
        self._iteration += 1
        h = hashlib.sha256(output.encode()).hexdigest()[:16]
        self._output_hashes.append(h)
        self._subtask_finger.append(subtask_id)

    def neuer_file_write(self) -> None:
        self._letzter_write = self._iteration

    def ist_loop(self) -> tuple[bool, str]:
        """Gibt (ist_loop, grund) zurück."""
        # 1. Output-Hash-Duplikat
        hashes = list(self._output_hashes)
        if len(hashes) >= 3 and len(set(hashes[-3:])) == 1:
            return True, "Gleicher Output dreimal hintereinander"

        # 2. Stagnation (kein File-Write)
        if (self._iteration - self._letzter_write) >= self.STAGNATION_LIMIT:
            return True, f"Keine Datei in {self.STAGNATION_LIMIT} Iterationen geschrieben"

        # 3. Subtask-Fingerprint-Loop
        fingers = list(self._subtask_finger)
        if len(fingers) >= 4 and len(set(fingers)) == 1:
            return True, "Gleicher Subtask 4x ohne Fortschritt"

        return False, ""
