"""Nova Predator v1 — SkillRegistry.
Hook-System, SkillContext/Result Contracts (frozen dataclasses).
Kein Nova-Import außer logger.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping
from core.logger import get

log = get("skill_registry")


class Hook(str, Enum):
    ON_MESSAGE = "on_message"
    ON_STARTUP = "on_startup"
    ON_RESPONSE = "on_response"
    ON_SCHEDULED = "on_scheduled"


@dataclass(frozen=True)
class SkillContext:
    """Frozen — Skills können context nicht mutieren."""
    user_input: str
    keywords: tuple[str, ...]
    intents: tuple[str, ...]
    brain_hits: tuple[Any, ...]        # tuple[BrainEntry, ...]
    session_facts: tuple[Any, ...]     # tuple[SessionFact, ...]
    skill_config: Mapping[str, Any]    # Skill-spezifische Config aus skill.yaml
    config: Mapping[str, Any]          # Globale Nova-Config


@dataclass(frozen=True)
class SkillResult:
    """Frozen — unveränderliches Ergebnis."""
    inhalt: str
    typ: str = "info"                  # 'info' | 'warnung' | 'fehler' | 'daten'
    metadaten: Mapping[str, Any] = field(default_factory=dict)


SkillFn = Callable[[SkillContext], SkillResult | None]


@dataclass
class SkillRegistrierung:
    name: str
    hook: Hook
    threshold: float
    funktion: SkillFn
    config: dict[str, Any] = field(default_factory=dict)
    aktiv: bool = True


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, list[SkillRegistrierung]] = {}

    def registriere(self, reg: SkillRegistrierung) -> None:
        key = reg.name
        if key not in self._skills:
            self._skills[key] = []
        # Existierenden Hook ersetzen falls vorhanden
        self._skills[key] = [
            r for r in self._skills[key] if r.hook != reg.hook
        ]
        self._skills[key].append(reg)
        log.debug("Skill registriert: %s/%s", reg.name, reg.hook.value)

    def entferne(self, name: str) -> None:
        self._skills.pop(name, None)
        log.info("Skill entfernt: %s", name)

    def deaktiviere(self, name: str) -> bool:
        if name not in self._skills:
            return False
        for r in self._skills[name]:
            r.aktiv = False
        return True

    def aktiviere(self, name: str) -> bool:
        if name not in self._skills:
            return False
        for r in self._skills[name]:
            r.aktiv = True
        return True

    def get_hooks(self, hook: Hook) -> list[SkillRegistrierung]:
        """Gibt alle aktiven Registrierungen für einen Hook zurück."""
        result = []
        for regs in self._skills.values():
            for r in regs:
                if r.hook == hook and r.aktiv:
                    result.append(r)
        return result

    def get_skill(self, name: str) -> list[SkillRegistrierung]:
        return self._skills.get(name, [])

    def alle_namen(self) -> list[str]:
        return list(self._skills.keys())

    def status(self) -> dict:
        return {
            "skills": len(self._skills),
            "aktiv": sum(1 for regs in self._skills.values() for r in regs if r.aktiv),
            "namen": self.alle_namen(),
        }
