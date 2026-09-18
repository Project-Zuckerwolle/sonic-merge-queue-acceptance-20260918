"""Nova Predator v1 — TaskManager.
Zentrale Verwaltung aller asyncio.Tasks — verhindert Task-Leaks (v9-Bug).
Python 3.14: asyncio.get_running_loop() statt get_event_loop().
Kein Nova-Import außer logger.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Coroutine

from core.logger import get

log = get("task_manager")


class TaskStatus(str, Enum):
    LAUFEND = "laufend"
    FERTIG = "fertig"
    FEHLER = "fehler"
    STORNIERT = "storniert"


@dataclass
class TaskInfo:
    name: str
    gestartet: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: TaskStatus = TaskStatus.LAUFEND
    fehler: str | None = None
    task: asyncio.Task | None = field(default=None, repr=False)


class TaskManager:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskInfo] = {}

    async def starte(
        self,
        name: str,
        coro: Coroutine[Any, Any, Any],
        timeout_s: float | None = None,
        ersetzen: bool = False,
    ) -> asyncio.Task:
        """Startet eine Coroutine als benannten Task.

        Args:
            name: Eindeutiger Task-Name
            coro: Coroutine die ausgeführt werden soll
            timeout_s: Optional — Task wird nach dieser Zeit abgebrochen
            ersetzen: Wenn True, wird ein laufender Task mit gleichem Namen gestoppt
        """
        # Existierenden Task behandeln
        if name in self._tasks:
            info = self._tasks[name]
            if info.task and not info.task.done():
                if ersetzen:
                    info.task.cancel()
                    log.debug("Task '%s' ersetzt", name)
                else:
                    log.debug("Task '%s' läuft bereits, übersprungen", name)
                    coro.close()  # Coroutine schließen um Warnung zu vermeiden
                    return info.task

        async def _wrapper() -> None:
            try:
                if timeout_s:
                    await asyncio.wait_for(coro, timeout=timeout_s)
                else:
                    await coro
                if name in self._tasks:
                    self._tasks[name].status = TaskStatus.FERTIG
            except asyncio.CancelledError:
                if name in self._tasks:
                    self._tasks[name].status = TaskStatus.STORNIERT
                raise
            except asyncio.TimeoutError:
                if name in self._tasks:
                    self._tasks[name].status = TaskStatus.FEHLER
                    self._tasks[name].fehler = f"Timeout nach {timeout_s}s"
                log.warning("Task '%s' Timeout nach %.1fs", name, timeout_s or 0)
            except Exception as e:
                if name in self._tasks:
                    self._tasks[name].status = TaskStatus.FEHLER
                    self._tasks[name].fehler = str(e)
                log.error("Task '%s' Fehler: %s", name, e)

        task = asyncio.create_task(_wrapper(), name=name)
        self._tasks[name] = TaskInfo(name=name, task=task)
        log.debug("Task gestartet: %s", name)
        return task

    async def storniere(self, name: str) -> bool:
        if name not in self._tasks:
            return False
        info = self._tasks[name]
        if info.task and not info.task.done():
            info.task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(info.task), timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            info.status = TaskStatus.STORNIERT
            return True
        return False

    async def alle_stoppen(self) -> None:
        """Stoppt alle laufenden Tasks — für Shutdown/Sleep."""
        namen = [n for n, i in self._tasks.items() if i.task and not i.task.done()]
        for name in namen:
            await self.storniere(name)
        log.info("Alle Tasks gestoppt (%d)", len(namen))

    def ist_aktiv(self, name: str) -> bool:
        info = self._tasks.get(name)
        return bool(info and info.task and not info.task.done())

    def status(self) -> dict:
        return {
            name: {
                "status": info.status.value,
                "gestartet": info.gestartet,
                "fehler": info.fehler,
            }
            for name, info in self._tasks.items()
        }

    def bereinige_fertige(self) -> int:
        """Entfernt abgeschlossene Tasks aus dem Dict."""
        fertige = [n for n, i in self._tasks.items() if i.task and i.task.done()]
        for n in fertige:
            del self._tasks[n]
        return len(fertige)
