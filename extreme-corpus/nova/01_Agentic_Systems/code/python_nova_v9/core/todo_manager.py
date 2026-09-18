"""Nova v8 – TodoManager.

Persistente Todo-Liste mit Priority-Levels (hoch/mittel/niedrig).
"""
from __future__ import annotations
import json
import uuid
from datetime import datetime
from pathlib import Path

from core.logger import get

log = get("todos")

PRIORITAETEN = {"hoch", "mittel", "niedrig"}


class TodoManager:
    def __init__(self, pfad: str = "./data/todos.json") -> None:
        self._pfad = Path(pfad)
        self._pfad.parent.mkdir(parents=True, exist_ok=True)
        self._todos: list[dict] = []
        self._laden()

    def _laden(self) -> None:
        if self._pfad.exists():
            try:
                self._todos = json.loads(self._pfad.read_text(encoding="utf-8"))
            except Exception:
                self._todos = []

    def _speichern(self) -> None:
        try:
            self._pfad.write_text(
                json.dumps(self._todos, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            log.error(f"Todo-Speicherfehler: {e}")

    def erstellen(self, text: str, kontext: str = "", prioritaet: str = "mittel") -> dict:
        prio = prioritaet if prioritaet in PRIORITAETEN else "mittel"
        todo = {
            "id":         str(uuid.uuid4())[:8],
            "text":       text,
            "kontext":    kontext,
            "prioritaet": prio,
            "erledigt":   False,
            "erstellt":   datetime.now().isoformat(),
            "erledigt_am": None,
        }
        self._todos.append(todo)
        self._speichern()
        return todo

    def erledigen(self, todo_id: str) -> bool:
        for t in self._todos:
            if t["id"] == todo_id:
                t["erledigt"]    = True
                t["erledigt_am"] = datetime.now().isoformat()
                self._speichern()
                return True
        return False

    def loeschen(self, todo_id: str) -> bool:
        vor = len(self._todos)
        self._todos = [t for t in self._todos if t["id"] != todo_id]
        if len(self._todos) < vor:
            self._speichern()
            return True
        return False

    def alle(self) -> list[dict]:
        return list(self._todos)

    def offen(self, prioritaet: str | None = None) -> list[dict]:
        result = [t for t in self._todos if not t["erledigt"]]
        if prioritaet:
            result = [t for t in result if t["prioritaet"] == prioritaet]
        prio_order = {"hoch": 0, "mittel": 1, "niedrig": 2}
        return sorted(result, key=lambda t: prio_order.get(t["prioritaet"], 1))

    def stats(self) -> dict:
        gesamt = len(self._todos)
        erledigt = sum(1 for t in self._todos if t["erledigt"])
        return {
            "gesamt": gesamt,
            "erledigt": erledigt,
            "offen": gesamt - erledigt,
            "hoch": sum(1 for t in self._todos if not t["erledigt"] and t.get("prioritaet") == "hoch"),
        }
