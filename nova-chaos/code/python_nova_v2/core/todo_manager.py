"""Nova Predator v3.3 — TodoManager.
CRUD Todos mit Prioritäten und Projekt-Tracking.
Kein Nova-Import außer logger.

v3.3:
  - Todo.projekt: optional — verknüpft Todo mit einem Projekt-Namen
  - Todo.notizen: Freitext-Feld für Projekt-Stand, Fortschritt
  - Todo.typ: 'aufgabe' | 'projekt_schritt' | 'projekt'
  - aktualisieren(): ändert Text/Notizen/Priorität ohne ID-Wechsel
  - projekte(): alle offenen Todos eines Projekts
"""
from __future__ import annotations
import json
import uuid
import dataclasses
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import aiofiles

from core.logger import get

log = get("todo_manager")

Prioritaet = Literal["hoch", "mittel", "niedrig"]
TodoTyp     = Literal["aufgabe", "projekt_schritt", "projekt"]


@dataclass
class Todo:
    id: str
    text: str
    prioritaet: Prioritaet = "mittel"
    erledigt: bool = False
    erstellt: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    erledigt_am: str | None = None
    # v3.3: Projekt-Felder
    typ: TodoTyp = "aufgabe"
    projekt: str | None = None          # Projekt-Name (z.B. "Nova v3.3")
    notizen: str = ""                   # Stand/Fortschritt Freitext
    aktualisiert: str | None = None     # Letztes Update-Datum

    def zu_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def von_dict(cls, d: dict) -> "Todo":
        erlaubte = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in erlaubte})


class TodoManager:
    def __init__(self, datei: str | Path = "data/todos.json") -> None:
        self._datei = Path(datei)
        self._todos: dict[str, Todo] = {}

    async def laden(self) -> None:
        if not self._datei.exists():
            return
        async with aiofiles.open(self._datei, encoding="utf-8") as f:
            daten = json.loads(await f.read())
        self._todos = {d["id"]: Todo.von_dict(d) for d in daten}
        log.debug("Todos geladen: %d", len(self._todos))

    async def add(
        self,
        text: str,
        prioritaet: Prioritaet = "mittel",
        typ: TodoTyp = "aufgabe",
        projekt: str | None = None,
        notizen: str = "",
    ) -> Todo:
        todo = Todo(
            id=str(uuid.uuid4()),
            text=text,
            prioritaet=prioritaet,
            typ=typ,
            projekt=projekt,
            notizen=notizen,
        )
        self._todos[todo.id] = todo
        await self._speichern()
        return todo

    async def aktualisieren(
        self,
        todo_id: str,
        text: str | None = None,
        notizen: str | None = None,
        prioritaet: Prioritaet | None = None,
    ) -> bool:
        """Aktualisiert einen bestehenden Todo ohne ID-Wechsel."""
        if todo_id not in self._todos:
            return False
        t = self._todos[todo_id]
        if text is not None:
            t.text = text
        if notizen is not None:
            t.notizen = notizen
        if prioritaet is not None:
            t.prioritaet = prioritaet
        t.aktualisiert = datetime.now(timezone.utc).isoformat()
        await self._speichern()
        return True

    async def erledigen(self, todo_id: str) -> bool:
        if todo_id not in self._todos:
            return False
        self._todos[todo_id].erledigt = True
        self._todos[todo_id].erledigt_am = datetime.now(timezone.utc).isoformat()
        await self._speichern()
        return True

    async def loeschen(self, todo_id: str) -> bool:
        if todo_id not in self._todos:
            return False
        del self._todos[todo_id]
        await self._speichern()
        return True

    def alle(self, nur_offen: bool = False) -> list[Todo]:
        todos = list(self._todos.values())
        if nur_offen:
            todos = [t for t in todos if not t.erledigt]
        return sorted(todos, key=lambda t: t.erstellt, reverse=True)

    def projekte(self, nur_offen: bool = True) -> dict[str, list[Todo]]:
        """Gibt alle Todos gruppiert nach Projekt-Namen zurück."""
        result: dict[str, list[Todo]] = {}
        for t in self._todos.values():
            if nur_offen and t.erledigt:
                continue
            if t.projekt:
                result.setdefault(t.projekt, []).append(t)
        return result

    def nach_projekt(self, projekt: str, nur_offen: bool = True) -> list[Todo]:
        """Alle Todos eines Projekts."""
        return [
            t for t in self._todos.values()
            if t.projekt == projekt and (not nur_offen or not t.erledigt)
        ]

    async def _speichern(self) -> None:
        self._datei.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(self._datei, "w", encoding="utf-8") as f:
            await f.write(json.dumps(
                [t.zu_dict() for t in self._todos.values()],
                ensure_ascii=False, indent=2,
            ))

