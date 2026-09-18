"""Nova Predator v3.1 — EventBus.
Async Pub/Sub. Kein Nova-Import außer logger.
Python 3.14: asyncio.get_running_loop() statt get_event_loop() (deprecated).
"""
from __future__ import annotations
import asyncio
from collections import defaultdict
from enum import Enum, auto
from typing import Any, Callable, Coroutine

from core.logger import get

log = get("event_bus")


class EventTyp(str, Enum):
    SERVER_START = "server.start"
    SERVER_STOP = "server.stop"
    NACHRICHT_EINGANG = "nachricht.eingang"
    NACHRICHT_FERTIG = "nachricht.fertig"
    SKILL_INSTALLIERT = "skill.installiert"
    SKILL_FEHLER = "skill.fehler"
    THINKER_START = "thinker.start"
    THINKER_FERTIG = "thinker.fertig"
    BRAIN_ENTRY_NEU = "brain.entry.neu"
    BRAIN_UPDATE = "brain.update"
    BREAKTHROUGH_IDEA = "brain.durchbruch"
    SCHEDULED_TASK = "scheduler.task"
    SCHLAF_BEGINN = "schlaf.beginn"
    SCHLAF_ENDE = "schlaf.ende"
    # A2A-Events (Nova Predator v2)
    A2A_HANDOFF = "a2a.handoff"
    A2A_RESULT = "a2a.result"
    ORCHESTRATOR_PLAN = "orchestrator.plan"
    SUBTASK_START = "subtask.start"
    SUBTASK_DONE = "subtask.done"
    SUBTASK_FAILED = "subtask.failed"


Handler = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class EventBus:
    def __init__(self) -> None:
        self._handler: dict[str, list[Handler]] = defaultdict(list)

    def abonniere(self, typ: EventTyp | str, handler: Handler) -> None:
        key = typ.value if isinstance(typ, EventTyp) else typ
        self._handler[key].append(handler)
        log.debug("Abonniert: %s → %s", key, getattr(handler, "__name__", "handler"))

    def abbestelle(self, typ: EventTyp | str, handler: Handler) -> None:
        key = typ.value if isinstance(typ, EventTyp) else typ
        vorher = len(self._handler[key])
        self._handler[key] = [h for h in self._handler[key] if h is not handler]
        if len(self._handler[key]) < vorher:
            log.debug("Abbestellt: %s → %s", key, getattr(handler, "__name__", "handler"))

    # English aliases (nova_ws.py and external code uses these)
    def subscribe(self, typ: EventTyp | str, handler: Handler) -> None:
        self.abonniere(typ, handler)

    def unsubscribe(self, typ: EventTyp | str, handler: Handler) -> None:
        self.abbestelle(typ, handler)

    async def publish(self, typ: EventTyp | str, daten: dict[str, Any] | None = None) -> None:
        key = typ.value if isinstance(typ, EventTyp) else typ
        payload = daten or {}
        handler_liste = list(self._handler.get(key, []))
        if not handler_liste:
            log.debug("Publish %s (keine Abonnenten)", key)
            return
        log.debug("Publish %s → %d Handler", key, len(handler_liste))
        # Alle Handler parallel, Fehler einzeln loggen ohne andere zu stoppen
        ergebnisse = await asyncio.gather(
            *[h(payload) for h in handler_liste],
            return_exceptions=True,
        )
        for i, r in enumerate(ergebnisse):
            if isinstance(r, Exception):
                log.error(
                    "Handler %s Fehler bei Event %s: %s",
                    handler_liste[i].__name__,
                    key,
                    r,
                )


# Globale Instanz
bus = EventBus()
