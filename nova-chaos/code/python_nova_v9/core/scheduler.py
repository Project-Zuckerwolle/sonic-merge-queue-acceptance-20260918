"""Nova v8 – Scheduler.

Async-Cron: regelmäßige Hintergrundaufgaben ohne externe Abhängigkeiten.
Konfigurierbar über config.yaml.
"""
from __future__ import annotations
import asyncio
from datetime import datetime, timedelta
from typing import Callable, Coroutine, Any

from core.logger import get
from core.event_bus import bus, EventTyp

log = get("scheduler")


class ScheduledTask:
    def __init__(
        self,
        name:          str,
        coro_factory:  Callable[[], Coroutine[Any, Any, None]],
        interval_s:    float | None = None,   # Intervall in Sekunden
        uhrzeit:       str   | None = None,   # "HH:MM" für tägliche Tasks
        einmalig:      bool         = False,
    ) -> None:
        self.name         = name
        self.coro_factory = coro_factory
        self.interval_s   = interval_s
        self.uhrzeit      = uhrzeit
        self.einmalig     = einmalig
        self.letzter_lauf: datetime | None = None
        self.laeufe:       int = 0
        self.fehler:       int = 0

    def naechster_lauf(self) -> datetime:
        jetzt = datetime.now()
        if self.uhrzeit:
            h, m = map(int, self.uhrzeit.split(":"))
            naechster = jetzt.replace(hour=h, minute=m, second=0, microsecond=0)
            if naechster <= jetzt:
                naechster += timedelta(days=1)
            return naechster
        if self.interval_s and self.letzter_lauf:
            return self.letzter_lauf + timedelta(seconds=self.interval_s)
        return jetzt


class Scheduler:
    def __init__(self) -> None:
        self._tasks:  list[ScheduledTask] = []
        self._lauft   = False
        self._task_handle: asyncio.Task | None = None

    def registriere(
        self,
        name:         str,
        coro_factory: Callable[[], Coroutine],
        interval_s:   float | None = None,
        uhrzeit:      str   | None = None,
        einmalig:     bool         = False,
    ) -> None:
        task = ScheduledTask(name, coro_factory, interval_s, uhrzeit, einmalig)
        self._tasks.append(task)
        log.info(f"Scheduler: '{name}' registriert (interval={interval_s if interval_s else '---'}s, uhrzeit={uhrzeit if uhrzeit else '---'})")

    async def starten(self) -> None:
        self._lauft = True
        self._task_handle = asyncio.create_task(self._loop())
        log.info(f"Scheduler gestartet: {len(self._tasks)} Tasks")

    async def stoppen(self) -> None:
        self._lauft = False
        if self._task_handle:
            self._task_handle.cancel()
            try:
                await self._task_handle
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._lauft:
            jetzt = datetime.now()
            for task in list(self._tasks):
                if self._soll_laufen(task, jetzt):
                    asyncio.create_task(self._fuehre_aus(task))
                    if task.einmalig:
                        self._tasks.remove(task)
            await asyncio.sleep(30)   # alle 30s prüfen

    def _soll_laufen(self, task: ScheduledTask, jetzt: datetime) -> bool:
        if task.uhrzeit:
            h, m = map(int, task.uhrzeit.split(":"))
            if jetzt.hour == h and jetzt.minute == m:
                if task.letzter_lauf is None or (jetzt - task.letzter_lauf).total_seconds() > 60:
                    return True
            return False
        if task.interval_s:
            if task.letzter_lauf is None:
                return True
            return (jetzt - task.letzter_lauf).total_seconds() >= task.interval_s
        return False

    async def _fuehre_aus(self, task: ScheduledTask) -> None:
        task.letzter_lauf = datetime.now()
        task.laeufe += 1
        log.info(f"Scheduler: starte '{task.name}' (Lauf #{task.laeufe})")
        try:
            await task.coro_factory()
            await bus.publish(EventTyp.SCHEDULED_TASK, {"name": task.name, "status": "ok"})
        except Exception as e:
            task.fehler += 1
            log.error(f"Scheduler: '{task.name}' Fehler: {e}", exc_info=True)
            await bus.publish(EventTyp.SCHEDULED_TASK, {"name": task.name, "status": "fehler", "fehler": str(e)})

    def jetzt_ausfuehren(self, name: str) -> bool:
        """Erzwingt sofortige Ausführung eines Tasks."""
        for task in self._tasks:
            if task.name == name:
                asyncio.create_task(self._fuehre_aus(task))
                return True
        return False

    def status(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "laeufe": t.laeufe,
                "fehler": t.fehler,
                "letzter_lauf": t.letzter_lauf.isoformat() if t.letzter_lauf else None,
                "naechster_lauf": t.naechster_lauf().isoformat(),
            }
            for t in self._tasks
        ]
