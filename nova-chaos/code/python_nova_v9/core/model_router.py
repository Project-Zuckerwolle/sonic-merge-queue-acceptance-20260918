"""Nova v8 – ModelRouter.

Wählt automatisch das passende Ollama-Modell je nach Aufgabentyp.
Verhindert dass jede Anfrage das große Chat-Modell belastet.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from core.logger import get
from core.config import get as cget

if TYPE_CHECKING:
    from core.ollama_client import OllamaClient

log = get("model_router")

# Aufgabentyp → Modell-Art
AUFGABEN_MODELL: dict[str, str] = {
    # Schnelle Aufgaben → kleines Modell
    "routing":     "schnell",
    "extraktion":  "schnell",
    "klassifikation": "schnell",
    "zusammenfassung_kurz": "schnell",
    # Code → Code-Modell
    "code":        "code",
    "debug":       "code",
    "refactor":    "code",
    # Alles andere → Chat-Modell
    "chat":        "chat",
    "reasoning":   "chat",
    "kreativ":     "chat",
    "zusammenfassung_lang": "chat",
    "research":    "chat",
}


class ModelRouter:
    def __init__(self, ollama: "OllamaClient", cfg: dict) -> None:
        self._ollama = ollama
        self._cfg    = cfg
        self._verfuegbare: set[str] = set()
        self._geprueft = False

    def verfuegbare_pruefen(self) -> None:
        """Prüft welche Modelle tatsächlich in Ollama vorhanden sind."""
        alle = ["chat", "schnell", "code", "embed"]
        for art in alle:
            modell = self._ollama.modell(art)
            if self._ollama.modell_verfuegbar(modell):
                self._verfuegbare.add(art)
            else:
                log.debug(f"Modell '{modell}' ({art}) nicht verfügbar")
        self._geprueft = True
        log.info(f"Verfügbare Modell-Arten: {self._verfuegbare}")

    def waehle_modell(self, aufgabe: str, intents: list[str] | None = None) -> str:
        """
        Wählt die beste Modell-Art für die Aufgabe.
        Fällt auf 'chat' zurück wenn bevorzugtes Modell nicht verfügbar.
        """
        # Intent-basierte Auswahl
        if intents:
            if "code" in intents or "debug" in intents:
                aufgabe = "code"
            elif "suche" in intents or "research" in intents:
                aufgabe = "research"

        gewuenscht = AUFGABEN_MODELL.get(aufgabe, "chat")

        # Fallback wenn Modell nicht verfügbar
        if self._geprueft and gewuenscht not in self._verfuegbare:
            fallback = "chat" if "chat" in self._verfuegbare else next(iter(self._verfuegbare), "chat")
            log.debug(f"Modell-Fallback: {gewuenscht} → {fallback}")
            return fallback

        return gewuenscht

    def status(self) -> dict:
        return {
            "verfuegbar": list(self._verfuegbare),
            "modelle": self._ollama.modelle_alle(),
            "geprueft": self._geprueft,
        }
