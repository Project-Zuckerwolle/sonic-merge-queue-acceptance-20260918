"""Nova Predator v1 — Persona.
Lädt persona.yaml und erzeugt den System-Prompt-Anteil für Ton/Verhalten.
Kein Nova-Import außer logger.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

import yaml

from core.logger import get

log = get("persona")

_STANDARD: dict[str, Any] = {
    "name": "Nova",
    "ton": "direkt und freundlich",
    "sprache": "Deutsch, Du-Form",
    "verboten": [],
    "verhalten": [],
}


class Persona:
    def __init__(self, pfad: str | Path = "persona.yaml") -> None:
        self._daten: dict[str, Any] = dict(_STANDARD)
        self._pfad = Path(pfad)
        self._laden()

    def _laden(self) -> None:
        if self._pfad.exists():
            with open(self._pfad, encoding="utf-8") as f:
                nutzer = yaml.safe_load(f) or {}
            self._daten.update(nutzer)
            log.debug("Persona geladen: %s", self._daten.get("name"))
        else:
            log.warning("persona.yaml nicht gefunden — Standard wird genutzt")

    def system_prompt(self, brain_fakten: list[str] | None = None) -> str:
        """Erzeugt den System-Prompt-Anteil für den LLM-Aufruf."""
        name = self._daten["name"]
        ton = self._daten["ton"]
        sprache = self._daten["sprache"]
        verboten = self._daten.get("verboten", [])
        verhalten = self._daten.get("verhalten", [])

        teile = [
            f"Du bist {name}, ein persönlicher KI-Assistent.",
            f"Ton: {ton}. Sprache: {sprache}.",
        ]

        if verhalten:
            teile.append("Verhalten:")
            for v in verhalten:
                teile.append(f"- {v}")

        if verboten:
            teile.append(f"Vermeide diese Phrasen: {', '.join(verboten)}")

        if brain_fakten:
            teile.append("\nBekannte Fakten (nutze sie aktiv, widersprich ihnen nicht):")
            for f_text in brain_fakten[:5]:
                teile.append(f"• {f_text}")

        return "\n".join(teile)

    @property
    def name(self) -> str:
        return self._daten.get("name", "Nova")

    def neu_laden(self) -> None:
        self._daten = dict(_STANDARD)
        self._laden()
