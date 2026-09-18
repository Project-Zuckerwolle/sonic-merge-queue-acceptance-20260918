"""Nova Predator v1 — ModelRouter.
Wählt das richtige LLM-Modell basierend auf Intent und Kontext.
Brain-Tasks   → qwen2.5:3b  (klein, schnell)
Chat-Tasks    → gemma4:e4b  (Hauptmodell)
Apex-Agent    → qwen3:30b-a3b  (MoE: 30B total, 3.3B aktiv — führendes Tool-Calling)
Coder-Tasks   → qwen3-coder:30b  (SWE-Bench optimiert, 256K Kontext)
Python 3.14 kompatibel: keine deprecated APIs.
"""
from __future__ import annotations
from typing import Literal

from core.logger import get

log = get("model_router")

# Modelltyp-Literal erweitert um "apex" und "coder"
Modelltyp = Literal["chat", "brain", "embed", "schnell", "apex", "coder"]


class ModelRouter:
    def __init__(
        self,
        chat_modell:   str = "gemma4:e4b",
        brain_modell:  str = "qwen2.5:3b",
        embed_modell:  str = "mxbai-embed-large",
        schnell_modell:str = "gemma4:e4b",
        apex_modell:   str = "qwen3:30b-a3b",
        coder_modell:  str = "qwen3-coder:30b",
    ) -> None:
        self._modelle: dict[str, str] = {
            "chat":    chat_modell,
            "brain":   brain_modell,
            "embed":   embed_modell,
            "schnell": schnell_modell,
            "apex":    apex_modell,
            "coder":   coder_modell,
        }

    def waehle(self, typ: Modelltyp) -> str:
        """Gibt den Modellnamen für den angegebenen Typ zurück.
        Fallback auf 'chat' wenn Typ unbekannt.
        """
        modell = self._modelle.get(typ, self._modelle["chat"])
        log.debug("ModelRouter: %s → %s", typ, modell)
        return modell

    def fuer_intent(self, intents: list[str]) -> str:
        """Wählt Modell basierend auf erkannten Intents.
        Code-Intent → Coder-Modell.
        Sonst → Chat-Modell.
        """
        if "code" in intents:
            return self._modelle.get("coder", self._modelle["chat"])
        return self._modelle["chat"]

    def alle(self) -> dict[str, str]:
        return dict(self._modelle)

    def aktualisiere(self, typ: str, modell: str) -> None:
        self._modelle[typ] = modell
        log.info("Modell '%s' aktualisiert: %s", typ, modell)
