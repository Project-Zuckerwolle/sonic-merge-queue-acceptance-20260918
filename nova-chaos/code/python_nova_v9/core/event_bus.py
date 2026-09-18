"""Nova v8 – Event Bus (Pub/Sub).

Zentrales Nervensystem: Module publishen Events, andere subscriben.
Unterstützt async und sync Callbacks. Thread-safe.

Nutzung:
    bus = EventBus()
    bus.subscribe(EventTyp.BRAIN_ENTRY_NEU, mein_callback)
    await bus.publish(EventTyp.BRAIN_ENTRY_NEU, {"id": "...", "titel": "..."})
"""
from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine

from core.logger import get

log = get("event_bus")


# ─── Event-Typen ──────────────────────────────────────────────────────────────
class EventTyp(Enum):
    # Brain
    BRAIN_ENTRY_NEU      = auto()
    BRAIN_ENTRY_UPDATE   = auto()
    BRAIN_ENTRY_LOESCHEN = auto()
    BRAIN_REINDEX        = auto()

    # Memory
    SESSION_NEU          = auto()
    SESSION_ARCHIV       = auto()
    MEMORY_HINZU         = auto()

    # Skills
    SKILL_INSTALLIERT    = auto()
    SKILL_FEHLER         = auto()
    SKILL_DEAKTIVIERT    = auto()
    SKILL_GESTARTET      = auto()
    SKILL_FERTIG         = auto()

    # Pipeline
    NACHRICHT_EINGANG    = auto()
    ANTWORT_FERTIG       = auto()
    PIPELINE_FEHLER      = auto()

    # Thinker
    THINKER_START        = auto()
    THINKER_FERTIG       = auto()
    THINKER_VERBINDUNG   = auto()

    # Claw / Agents
    AGENT_GESTARTET      = auto()
    AGENT_OUTPUT         = auto()
    AGENT_FERTIG         = auto()

    # Scheduler
    SCHEDULED_TASK       = auto()
    BRIEFING_GENERIERT   = auto()

    # System
    SERVER_START         = auto()
    SERVER_STOP          = auto()
    OLLAMA_OFFLINE       = auto()
    OLLAMA_ONLINE        = auto()

    # MCP
    MCP_VERBUNDEN        = auto()
    MCP_GETRENNT         = auto()

    # Workspace
    DATEI_NEU            = auto()
    DATEI_GEAENDERT      = auto()


@dataclass
class Event:
    typ: EventTyp
    daten: dict[str, Any] = field(default_factory=dict)
    quelle: str = "system"


# ─── Callback-Typen ───────────────────────────────────────────────────────────
AsyncCallback = Callable[[Event], Coroutine[Any, Any, None]]
SyncCallback  = Callable[[Event], None]
AnyCallback   = AsyncCallback | SyncCallback


# ─── Event Bus ────────────────────────────────────────────────────────────────
class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[EventTyp, list[AnyCallback]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """App-Loop registrieren (aus lifespan aufrufen)."""
        self._loop = loop

    def subscribe(self, typ: EventTyp, callback: AnyCallback) -> None:
        """Callback für Event-Typ registrieren."""
        self._subscribers.setdefault(typ, []).append(callback)

    def unsubscribe(self, typ: EventTyp, callback: AnyCallback) -> None:
        """Callback entfernen."""
        if typ in self._subscribers:
            self._subscribers[typ] = [
                cb for cb in self._subscribers[typ] if cb is not callback
            ]

    async def publish(self, typ: EventTyp, daten: dict | None = None, quelle: str = "system") -> None:
        """Event asynchron publishen – alle Subscriber werden benachrichtigt."""
        event = Event(typ=typ, daten=daten or {}, quelle=quelle)
        callbacks = self._subscribers.get(typ, [])
        if not callbacks:
            return
        for cb in callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)
            except Exception as e:
                log.error(f"EventBus Fehler ({typ.name} → {cb.__name__}): {e}", exc_info=True)

    def publish_threadsafe(self, typ: EventTyp, daten: dict | None = None, quelle: str = "system") -> None:
        """Event aus einem Thread publishen (thread-safe via call_soon_threadsafe)."""
        if self._loop is None or self._loop.is_closed():
            log.warning(f"EventBus: kein Loop für threadsafe publish ({typ.name})")
            return
        asyncio.run_coroutine_threadsafe(
            self.publish(typ, daten, quelle), self._loop
        )

    def subscriber_anzahl(self, typ: EventTyp) -> int:
        return len(self._subscribers.get(typ, []))

    def alle_typen(self) -> list[str]:
        return [t.name for t in self._subscribers if self._subscribers[t]]


# ─── Singleton ────────────────────────────────────────────────────────────────
bus = EventBus()
