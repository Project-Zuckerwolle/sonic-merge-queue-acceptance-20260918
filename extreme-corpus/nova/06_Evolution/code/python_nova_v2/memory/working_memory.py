"""Nova Predator v1 — WorkingMemory.
Aktueller Turn — reines RAM, kein Persist.
Hält user_input, q_vec, Pipeline-State des laufenden Requests.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkingMemory:
    """Lebt nur für einen Turn. Wird am Turn-Beginn neu gesetzt."""
    user_input: str = ""
    q_vec: list[float] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    aktive_skills: list[tuple[str, float]] = field(default_factory=list)
    skill_ergebnisse: dict[str, Any] = field(default_factory=dict)
    todo_signal: bool = False
    todo_text: str = ""
    dauer_ms: float = 0.0

    def reset(self) -> None:
        self.user_input = ""
        self.q_vec = []
        self.keywords = []
        self.intents = []
        self.aktive_skills = []
        self.skill_ergebnisse = {}
        self.todo_signal = False
        self.todo_text = ""
        self.dauer_ms = 0.0

    def als_dict(self) -> dict:
        return {
            "user_input": self.user_input[:100],
            "keywords": self.keywords[:10],
            "intents": self.intents,
            "aktive_skills": self.aktive_skills,
            "todo_signal": self.todo_signal,
            "dauer_ms": round(self.dauer_ms, 1),
        }
