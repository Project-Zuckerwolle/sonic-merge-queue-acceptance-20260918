"""Nova v9 – BrainWriter.

Asynchrone LLM-Faktenextraktion nach einem Gespraechsaustausch.
Schreibt neue, dauerhaft relevante Fakten ins Brain.
NOOP wenn keine neuen Fakten erkannt wurden.

v8-Fixes:
    - call_soon_threadsafe Lambda-Bug gefixt (Variable wurde nicht korrekt gecaptured)
    - BrainWriter laeuft als Background-Task, blockiert nicht den Chat
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

from core.logger    import get
from core.event_bus import bus, EventTyp

if TYPE_CHECKING:
    from core.ollama_client import OllamaClient
    from core.brain_manager import BrainManager

log = get("brain_writer")

_TYPEN = "person|fahrzeug|projekt|praeferenz|technik|termin|ort|finanzen|tier|gesundheit|notiz|wissen|sonstiges"

_PROMPT_TEMPLATE = (
    "Analysiere diesen Gespraechsaustausch und extrahiere neue, faktische "
    "Informationen ueber den Nutzer oder seine Welt.\n"
    "Antworte NUR mit JSON oder dem Wort NOOP (wenn keine neuen Fakten vorhanden).\n\n"
    'Format: {"entries": [{"titel": "...", "inhalt": "...", "typ": "' + _TYPEN + '", "tags": ["..."]}]}\n\n'
    "Regeln:\n"
    "- Nur neue, dauerhaft relevante Fakten (keine temporaeren Informationen)\n"
    "- Maximal 3 Eintraege pro Austausch\n"
    "- NOOP wenn: allgemeine Fragen, keine persoenlichen Infos, nur Smalltalk\n\n"
    "Austausch:\nUser: "
)


class BrainWriter:
    def __init__(self, ollama: "OllamaClient", brain: "BrainManager") -> None:
        self._ollama = ollama
        self._brain  = brain

    async def verarbeite(self, user_input: str, antwort: str) -> list[str]:
        """
        Asynchrone Faktenextraktion. Laeuft als Background-Task.
        Gibt Liste der erstellten Entry-IDs zurueck.
        """
        return await loop.run_in_executor(
            None,
            self._extrahiere_und_schreibe,
            user_input,
            antwort,
            loop,
        )

    def _extrahiere_und_schreibe(
        self,
        user_input: str,
        antwort:    str,
        loop:       asyncio.AbstractEventLoop,
    ) -> list[str]:
        prompt = (
            _PROMPT_TEMPLATE
            + user_input[:500]
            + "\nNova: "
            + antwort[:500]
        )
        roh = self._ollama.generiere(
            [{"role": "user", "content": prompt}],
            modell_art="schnell",
            max_tokens=400,
        )
        if not roh:
            return []

        roh_clean = roh.strip()
        if roh_clean.upper() == "NOOP" or roh_clean[:6].upper() == "NOOP":
            log.debug("BrainWriter: NOOP - keine neuen Fakten")
            return []

        eintraege = self._parse(roh_clean)
        erstellte: list[str] = []

        for e in eintraege[:3]:
            titel  = e.get("titel", "").strip()
            inhalt = e.get("inhalt", "").strip()
            typ    = e.get("typ", "sonstiges")
            tags   = e.get("tags", [])
            if not titel or not inhalt:
                continue

            entry_id = self._brain.eintrag_erstellen(titel, inhalt, tags, typ, tags)
            erstellte.append(entry_id)

            # Event-Bus benachrichtigen (thread-safe, kein Lambda-Bug)
            eid_captured = entry_id
            titel_captured = titel
            loop.call_soon_threadsafe(
                bus.publish_threadsafe,
                EventTyp.BRAIN_ENTRY_NEU,
                {"id": eid_captured, "titel": titel_captured},
            )

        if erstellte:
            log.info(f"BrainWriter: {len(erstellte)} neue Entries | {erstellte}")

        return erstellte

    def _parse(self, text: str) -> list[dict]:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group())
            return data.get("entries", []) or data.get("eintraege", [])
        except json.JSONDecodeError:
            return []
