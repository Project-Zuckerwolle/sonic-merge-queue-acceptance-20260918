"""Nova Predator v1 — ContextBuilder.
Baut den vollständigen LLM-Kontext in der richtigen Reihenfolge auf.
Token-Budget wird eingehalten. Fakten werden nie gekürzt.

Reihenfolge (unveränderlich):
  [1] Persona + System-Prompt
  [2] Brain-Fakten (top-5, vertrauen > 0.5)
  [3] Session-Fakten (diese Session)
  [4] Chat-History (rolling window, wird zuerst gekürzt)
  [5] Skill-Ergebnisse (wenn Budget erlaubt)
  [6] User-Input (immer zuletzt)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.logger import get

if TYPE_CHECKING:
    from core.persona import Persona
    from core.skill_registry import SkillResult
    from memory.episodic_memory import EpisodicMemory
    from memory.session_facts import SessionFacts

log = get("context_builder")


@dataclass
class KontextErgebnis:
    """Fertiger Kontext für einen LLM-Aufruf."""
    messages: list[dict[str, str]]
    system: str
    brain_fakten: list[str]
    session_fakten_count: int
    token_schaetzung: int
    auslastung: float
    brain_entry_ids: list[str] = field(default_factory=list)  # v3.3: für Feedback-Loop


class ContextBuilder:
    def __init__(
        self,
        persona: "Persona",
        episodic: "EpisodicMemory",
        session_facts: "SessionFacts",
        max_tokens: int = 8000,
    ) -> None:
        self._persona = persona
        self._episodic = episodic
        self._sf = session_facts
        self._max_tokens = max_tokens

    def baue(
        self,
        user_input: str,
        brain_fakten: list[str] | None = None,
        skill_ergebnisse: dict[str, "SkillResult"] | None = None,
    ) -> KontextErgebnis:
        """Baut den vollständigen Kontext auf."""
        brain_fakten = brain_fakten or []
        skill_ergebnisse = skill_ergebnisse or {}

        # [1] System-Prompt (Persona + Brain-Fakten)
        system = self._persona.system_prompt(brain_fakten)
        token_system = len(system) // 4

        # [3] Session-Fakten als extra System-Abschnitt
        sf_text = self._sf.als_kontext_text(min_konfidenz=0.7)
        if sf_text:
            system = system + "\n\n" + sf_text
            token_system += len(sf_text) // 4

        # Budget für History = max_tokens - system - user_input - skill - Puffer
        token_user = len(user_input) // 4 + 1
        token_skill = sum(len(r.inhalt) // 4 for r in skill_ergebnisse.values())
        token_puffer = 200

        token_fuer_history = (
            self._max_tokens - token_system - token_user - token_skill - token_puffer
        )
        token_fuer_history = max(500, token_fuer_history)

        # [4] Chat-History (rolling window)
        messages = self._episodic.als_llm_nachrichten(max_tokens=token_fuer_history)

        # [5] Skill-Ergebnisse einbauen (als system-Ergänzung wenn Budget reicht)
        if skill_ergebnisse:
            skill_texte = []
            for name, result in skill_ergebnisse.items():
                skill_texte.append(f"[{name}]: {result.inhalt}")
            skill_block = "\n\nAktuelle Skill-Ergebnisse:\n" + "\n".join(skill_texte)
            verbleibend = self._max_tokens - token_system - token_fuer_history - token_user
            if len(skill_block) // 4 <= verbleibend:
                system = system + skill_block

        # [6] User-Input als letzte Nachricht
        messages.append({"role": "user", "content": user_input})

        token_gesamt = token_system + sum(len(m["content"]) // 4 for m in messages)
        auslastung = min(1.0, token_gesamt / self._max_tokens)

        return KontextErgebnis(
            messages=messages,
            system=system,
            brain_fakten=brain_fakten,
            session_fakten_count=len(self._sf),
            token_schaetzung=token_gesamt,
            auslastung=auslastung,
        )
