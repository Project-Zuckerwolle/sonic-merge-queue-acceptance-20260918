"""Nova Predator v1 — SessionFacts.
Leichtgewichtiger Fact-Store innerhalb einer Session.
Fakten die DIESE Session extrahiert hat — schneller als Brain-Lookup.
Anti-Hallucination: LLM sieht diese Facts VOR der Antwort.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

FactTyp = Literal["fakt", "praeferenz", "kontext", "aufgabe"]


@dataclass(frozen=True)
class SessionFact:
    """Frozen — unveränderlich nach Erstellung."""
    text: str
    typ: FactTyp = "fakt"
    konfidenz: float = 0.9     # 0.0–1.0
    erstellt: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def als_text(self) -> str:
        return self.text


class SessionFacts:
    """Hält verifizierte Fakten der laufenden Session."""

    def __init__(self, max_fakten: int = 30) -> None:
        self._fakten: list[SessionFact] = []
        self._max = max_fakten

    def add(self, text: str, typ: FactTyp = "fakt", konfidenz: float = 0.9) -> SessionFact:
        # Duplikat-Prüfung (einfach: gleicher normierter Text)
        text_norm = text.strip().lower()
        for f in self._fakten:
            if f.text.strip().lower() == text_norm:
                return f   # Bereits vorhanden
        fact = SessionFact(text=text.strip(), typ=typ, konfidenz=konfidenz)
        self._fakten.append(fact)
        # Wenn zu viele: älteste niedrig-konfidente zuerst entfernen
        if len(self._fakten) > self._max:
            self._fakten.sort(key=lambda f: f.konfidenz, reverse=True)
            self._fakten = self._fakten[: self._max]
        return fact

    def alle(self, min_konfidenz: float = 0.0) -> list[SessionFact]:
        return [f for f in self._fakten if f.konfidenz >= min_konfidenz]

    def als_kontext_text(self, min_konfidenz: float = 0.7) -> str:
        fakten = self.alle(min_konfidenz)
        if not fakten:
            return ""
        zeilen = [f"• {f.text}" for f in fakten]
        return "Fakten aus dieser Session:\n" + "\n".join(zeilen)

    def als_tuple(self) -> tuple[SessionFact, ...]:
        return tuple(self._fakten)

    def reset(self) -> None:
        self._fakten.clear()

    def __len__(self) -> int:
        return len(self._fakten)
