"""Nova v8 – Working Memory.

Kurzzeitgedächtnis: aktiver Task-Kontext, Attention-Stack.
RAM-only, kein Disk-Zugriff — schnellste Ebene der Memory-Hierarchie.
Wird pro Konversationsschritt aktualisiert.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.logger import get

log = get("working_mem")

# Max Einträge im Attention-Stack
MAX_ATTENTION = 6
# Max aktive Tasks
MAX_TASKS = 10


@dataclass
class AttentionItem:
    inhalt: str
    quelle: str          # "user" | "assistant" | "skill" | "brain"
    relevanz: float = 1.0
    zeitstempel: datetime = field(default_factory=datetime.now)


@dataclass
class Task:
    id: str
    beschreibung: str
    status: str = "aktiv"    # aktiv | wartend | fertig | fehler
    erstellt: datetime = field(default_factory=datetime.now)
    ergebnis: Any = None
    metadaten: dict = field(default_factory=dict)


class WorkingMemory:
    """Kurzzeit-Gedächtnis für den aktuell verarbeiteten Kontext."""

    def __init__(self) -> None:
        self._attention: deque[AttentionItem] = deque(maxlen=MAX_ATTENTION)
        self._tasks: dict[str, Task] = {}
        self._kontext_vars: dict[str, Any] = {}  # schnelle KV für aktuelle Session
        self._aktuelle_nachricht: str = ""
        self._aktuelle_keywords: list[str] = []
        self._aktuelle_intents: list[str] = []

    # ─── Aktuelle Nachricht ───────────────────────────────────────────────────
    def setze_nachricht(self, text: str, keywords: list[str], intents: list[str]) -> None:
        self._aktuelle_nachricht = text
        self._aktuelle_keywords = keywords
        self._aktuelle_intents = intents
        log.debug(f"Nachricht gesetzt | {len(text)} Zeichen | keywords={keywords} | intents={intents}")

    @property
    def nachricht(self) -> str:
        return self._aktuelle_nachricht

    @property
    def keywords(self) -> list[str]:
        return self._aktuelle_keywords

    @property
    def intents(self) -> list[str]:
        return self._aktuelle_intents

    # ─── Attention Stack ─────────────────────────────────────────────────────
    def attention_hinzu(self, inhalt: str, quelle: str, relevanz: float = 1.0) -> None:
        self._attention.appendleft(
            AttentionItem(inhalt=inhalt, quelle=quelle, relevanz=relevanz)
        )
        log.debug(f"Attention hinzu | quelle={quelle} | relevanz={relevanz} | '{inhalt[:60]}'")

    def attention_kontext(self, max_items: int = MAX_ATTENTION) -> list[AttentionItem]:
        """Gibt aktuelle Attention-Items zurück, nach Relevanz sortiert."""
        items = list(self._attention)[:max_items]
        return sorted(items, key=lambda x: x.relevanz, reverse=True)

    def attention_als_text(self, max_items: int = 4) -> str:
        """Kompakter Text aller Attention-Items für LLM-Kontext."""
        items = self.attention_kontext(max_items)
        if not items:
            return ""
        lines = [f"[{i.quelle}] {i.inhalt[:200]}" for i in items]
        return "\n".join(lines)

    def attention_leeren(self) -> None:
        self._attention.clear()

    # ─── Tasks ────────────────────────────────────────────────────────────────
    def task_erstellen(self, task_id: str, beschreibung: str, metadaten: dict | None = None) -> Task:
        if len(self._tasks) >= MAX_TASKS:
            fertige = [k for k, t in self._tasks.items() if t.status in ("fertig", "fehler")]
            if fertige:
                del self._tasks[fertige[0]]
        task = Task(id=task_id, beschreibung=beschreibung, metadaten=metadaten or {})
        self._tasks[task_id] = task
        log.debug(f"Task erstellt | id={task_id} | '{beschreibung[:60]}'")
        return task

    def task_update(self, task_id: str, status: str, ergebnis: Any = None) -> None:
        if task_id in self._tasks:
            self._tasks[task_id].status = status
            if ergebnis is not None:
                self._tasks[task_id].ergebnis = ergebnis
            log.debug(f"Task update | id={task_id} | status={status}")

    def task_holen(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def aktive_tasks(self) -> list[Task]:
        return [t for t in self._tasks.values() if t.status == "aktiv"]

    # ─── Kontext-Variablen ────────────────────────────────────────────────────
    def setze(self, key: str, wert: Any) -> None:
        self._kontext_vars[key] = wert

    def hole(self, key: str, default: Any = None) -> Any:
        return self._kontext_vars.get(key, default)

    def entferne(self, key: str) -> None:
        self._kontext_vars.pop(key, None)

    # ─── Reset ────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Für neue Konversation / nach Session-Ende."""
        self._attention.clear()
        self._tasks = {k: t for k, t in self._tasks.items() if t.status == "aktiv"}
        self._kontext_vars.clear()
        self._aktuelle_nachricht = ""
        self._aktuelle_keywords = []
        self._aktuelle_intents = []

    # ─── Status ───────────────────────────────────────────────────────────────
    def status(self) -> dict:
        return {
            "attention_items": len(self._attention),
            "aktive_tasks": len(self.aktive_tasks()),
            "kontext_vars": list(self._kontext_vars.keys()),
        }
