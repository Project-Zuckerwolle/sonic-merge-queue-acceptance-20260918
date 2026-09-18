"""Nova Predator v1 — Scheduler.
Async-Cron für zeitgesteuerte Tasks.
Python 3.14: asyncio.get_running_loop() pattern.
Kein Nova-Import außer logger und task_manager.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, time
from typing import Any, Callable, Coroutine

from core.logger import get
from core.task_manager import TaskManager

log = get("scheduler")

CoroFactory = Callable[[], Coroutine[Any, Any, Any]]


@dataclass
class ScheduledTask:
    name: str
    fabrik: CoroFactory           # Callable der eine neue Coroutine erzeugt
    intervall_s: float | None = None   # Wiederholung alle N Sekunden
    uhrzeit: str | None = None         # "HH:MM" für tägliche Ausführung
    aktiv: bool = True
    letzter_run: str | None = None


class Scheduler:
    def __init__(self, task_manager: TaskManager) -> None:
        self._tm = task_manager
        self._tasks: dict[str, ScheduledTask] = {}
        self._lauf_task: asyncio.Task | None = None
        self._aktiv = False

    def registriere(
        self,
        name: str,
        fabrik: CoroFactory,
        intervall_s: float | None = None,
        uhrzeit: str | None = None,
    ) -> None:
        self._tasks[name] = ScheduledTask(
            name=name,
            fabrik=fabrik,
            intervall_s=intervall_s,
            uhrzeit=uhrzeit,
        )
        log.debug(
            "Scheduler-Task registriert: %s (intervall=%s, uhrzeit=%s)",
            name, intervall_s, uhrzeit,
        )

    def deregistriere(self, name: str) -> None:
        self._tasks.pop(name, None)

    async def starten(self) -> None:
        self._aktiv = True
        self._lauf_task = asyncio.create_task(self._schleife(), name="scheduler")
        log.info("Scheduler gestartet (%d Tasks)", len(self._tasks))

    async def stoppen(self) -> None:
        self._aktiv = False
        if self._lauf_task and not self._lauf_task.done():
            self._lauf_task.cancel()
            try:
                await self._lauf_task
            except asyncio.CancelledError:
                pass
        log.info("Scheduler gestoppt")

    async def manuell_ausfuehren(self, name: str) -> bool:
        if name not in self._tasks or not self._tasks[name].aktiv:
            return False
        task = self._tasks[name]
        await self._tm.starte(f"scheduled_{name}", task.fabrik(), ersetzen=True)
        task.letzter_run = datetime.now(timezone.utc).isoformat()
        return True

    async def _schleife(self) -> None:
        while self._aktiv:
            await asyncio.sleep(30)   # Alle 30s prüfen
            jetzt = datetime.now(timezone.utc)
            for task in self._tasks.values():
                if not task.aktiv:
                    continue
                soll_laufen = False

                if task.intervall_s:
                    if task.letzter_run is None:
                        soll_laufen = True
                    else:
                        letzter = datetime.fromisoformat(task.letzter_run)
                        vergangen = (jetzt - letzter).total_seconds()
                        soll_laufen = vergangen >= task.intervall_s

                elif task.uhrzeit:
                    ziel_h, ziel_m = map(int, task.uhrzeit.split(":"))
                    if jetzt.hour == ziel_h and jetzt.minute == ziel_m:
                        if task.letzter_run is None:
                            soll_laufen = True
                        else:
                            letzter = datetime.fromisoformat(task.letzter_run)
                            # Nicht zweimal in derselben Minute
                            soll_laufen = (jetzt - letzter).total_seconds() > 60

                if soll_laufen:
                    log.info("Scheduler führt aus: %s", task.name)
                    await self._tm.starte(
                        f"scheduled_{task.name}",
                        task.fabrik(),
                        ersetzen=True,
                    )
                    task.letzter_run = jetzt.isoformat()

    def status(self) -> dict:
        return {
            name: {
                "aktiv": t.aktiv,
                "intervall_s": t.intervall_s,
                "uhrzeit": t.uhrzeit,
                "letzter_run": t.letzter_run,
            }
            for name, t in self._tasks.items()
        }
