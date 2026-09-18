"""Nova Predator v3.3 — UserMdUpdater.

Nach jedem Chat-Turn entscheidet das Nano-LLM ob der Turn neue
persistente User-Infos enthält. Wenn ja, wird behavior/USER.md
sofort aktualisiert und persona.neu_laden() aufgerufen.

Was gehört in USER.md:
  - Name, Wohnort, Beruf, System-Setup des Users
  - Aktive Projekte + ihr aktueller Stand
  - Präferenzen (Tools, Sprachen, Workflows)
  - Wichtige persönliche Fakten die Nova langfristig kennen soll

Was NICHT rein gehört:
  - Allgemeines Wissen / Smalltalk
  - Temporäre Aussagen
  - Dinge die Nova gerade erklärt hat

Design:
  - LLM-Urteil ist binär: "hat neue User-Info" ja/nein
  - Wenn ja: zweiter Call generiert nur den neuen Abschnitt
  - Merge in bestehende USER.md (kein Vollrewrite bei jedem Turn)
  - asyncio.Lock gegen parallele Schreibzugriffe
  - Fire-and-forget, blockiert nie den Chat-Flow
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from core.logger import get

if TYPE_CHECKING:
    from core.ollama_client import OllamaClient
    from core.persona import Persona

log = get("user_md_updater")

_USER_MD_PFAD = Path("behavior/USER.md")

_CHECK_PROMPT = """\
Analysiere diesen Gesprächs-Turn:
USER: {user_input}
NOVA: {antwort}

Enthält er neue persistente Informationen ÜBER DEN USER selbst?
(Name, System, Projekte, Präferenzen, persönliche Fakten — nichts temporäres)

Antworte NUR mit: JA oder NEIN
"""

_EXTRAKT_PROMPT = """\
Extrahiere die neuen User-Informationen aus diesem Turn:
USER: {user_input}
NOVA: {antwort}

Bereits bekannt (USER.md):
{user_md_inhalt}

Schreibe NUR neue/aktualisierte Infos als Markdown (## Abschnitt + Bullets).
Wenn nichts wirklich neu: antworte mit NICHTS_NEU
Keine Erklärungen, nur Fakten.
"""

_MERGE_PROMPT = """\
Merge neue Infos in die bestehende USER.md.

Bestehend:
{bestehend}

Neu:
{neu}

Regeln: Bestehende gültige Infos behalten. Überschriebenes aktualisieren.
Platzhalter "(noch leer)" entfernen. Struktur (## Abschnitte) behalten.
Antworte NUR mit dem fertigen USER.md Inhalt, kein Kommentar.
"""


class UserMdUpdater:
    """Aktualisiert USER.md nach jedem Turn — LLM entscheidet ob nötig."""

    def __init__(
        self,
        ollama: "OllamaClient",
        persona: "Persona",
        modell: str = "qwen2.5:3b",
        user_md_pfad: str | Path = _USER_MD_PFAD,
    ) -> None:
        self._ollama  = ollama
        self._persona = persona
        self._modell  = modell
        self._pfad    = Path(user_md_pfad)
        self._lock    = asyncio.Lock()
        self._letzter_update: str | None = None

    async def nach_turn_pruefen(self, user_input: str, antwort: str) -> bool:
        """Prüft ob Turn neue User-Info enthält und updatet USER.md.
        Fire-and-forget geeignet — wirft keine Exceptions.
        """
        if len(user_input.strip()) < 8:
            return False
        try:
            return await self._update_wenn_noetig(user_input, antwort)
        except Exception as e:
            log.debug("UserMdUpdater Fehler: %s", e)
            return False

    async def _update_wenn_noetig(self, user_input: str, antwort: str) -> bool:
        # Schritt 1: Braucht es ein Update?
        entscheidung = await self._ollama.chat(
            nachrichten=[{"role": "user", "content": _CHECK_PROMPT.format(
                user_input=user_input[:400],
                antwort=antwort[:400],
            )}],
            modell=self._modell,
            optionen={"temperature": 0.0, "num_predict": 5, "think": False},
        )
        if "JA" not in entscheidung.strip().upper():
            return False

        log.info("UserMdUpdater: Turn enthält User-Info — update USER.md")

        # Schritt 2: Neuen Extrakt generieren
        user_md_aktuell = self._pfad.read_text(encoding="utf-8") \
            if self._pfad.exists() else ""

        neuer_extrakt = await self._ollama.chat(
            nachrichten=[{"role": "user", "content": _EXTRAKT_PROMPT.format(
                user_input=user_input[:600],
                antwort=antwort[:600],
                user_md_inhalt=user_md_aktuell[:1500],
            )}],
            modell=self._modell,
            optionen={"temperature": 0.1, "num_predict": 400, "think": False},
        )
        neuer_extrakt = neuer_extrakt.strip()

        if not neuer_extrakt or "NICHTS_NEU" in neuer_extrakt:
            return False

        # Schritt 3: Merge + schreiben (Lock gegen Parallelzugriff)
        async with self._lock:
            await self._merge_und_schreiben(user_md_aktuell, neuer_extrakt)

        # Schritt 4: Persona sofort neu laden
        self._persona.neu_laden()
        self._letzter_update = datetime.now(timezone.utc).isoformat()
        log.info("USER.md aktualisiert + Persona neu geladen")
        return True

    async def _merge_und_schreiben(self, bestehend: str, neu: str) -> None:
        # Erster echter Eintrag: direkt schreiben
        hat_inhalt = any(
            z.strip() and not z.strip().startswith("#")
            and "(noch leer" not in z and "(leer)" not in z and z.strip() != "---"
            for z in bestehend.splitlines()
        )

        if not hat_inhalt:
            inhalt = f"# USER.md — Nova Predator\n\n{neu}\n"
        else:
            merged = await self._ollama.chat(
                nachrichten=[{"role": "user", "content": _MERGE_PROMPT.format(
                    bestehend=bestehend[:2000],
                    neu=neu,
                )}],
                modell=self._modell,
                optionen={"temperature": 0.0, "num_predict": 800, "think": False},
            )
            inhalt = merged.strip() or f"{bestehend}\n\n{neu}\n"

        self._pfad.parent.mkdir(parents=True, exist_ok=True)
        self._pfad.write_text(inhalt, encoding="utf-8")

    def status(self) -> dict:
        return {
            "letzter_update": self._letzter_update,
            "pfad": str(self._pfad),
            "existiert": self._pfad.exists(),
            "groesse": self._pfad.stat().st_size if self._pfad.exists() else 0,
        }
