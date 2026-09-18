"""Nova Predator v3.1 — Persona.
Lädt persona.yaml + behavior/USER.md und erzeugt den System-Prompt-Anteil.
Fügt bei jedem Aufruf aktuelles Datum/Uhrzeit und Tool-Hinweise hinzu.
Kein Nova-Import außer logger.
"""
from __future__ import annotations
from datetime import datetime
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

# Standard-Pfad für persistente User-Fakten
_USER_MD_PFAD = Path("behavior/USER.md")

# Platzhalter die in leerer USER.md stehen — werden ausgefiltert
_USER_MD_LEER_MARKER = ("(noch leer", "(leer)", "# USER.md", "# Nova Predator")

# Deutsche Wochentage für System-Prompt
_WOCHENTAGE = ["Montag", "Dienstag", "Mittwoch", "Donnerstag",
               "Freitag", "Samstag", "Sonntag"]


class Persona:
    def __init__(
        self,
        pfad: str | Path = "persona.yaml",
        user_md_pfad: str | Path | None = None,
    ) -> None:
        self._daten: dict[str, Any] = dict(_STANDARD)
        self._pfad = Path(pfad)
        self._user_md_pfad = Path(user_md_pfad) if user_md_pfad else _USER_MD_PFAD
        self._user_md_inhalt: str = ""
        self._laden()

    def _laden(self) -> None:
        """Lädt persona.yaml und behavior/USER.md."""
        # 1. persona.yaml
        if self._pfad.exists():
            with open(self._pfad, encoding="utf-8") as f:
                nutzer = yaml.safe_load(f) or {}
            self._daten.update(nutzer)
            log.debug("Persona geladen: %s", self._daten.get("name"))
        else:
            log.warning("persona.yaml nicht gefunden — Standard wird genutzt")

        # 2. behavior/USER.md — persistente User-Fakten
        self._user_md_inhalt = self._lade_user_md()

    def _lade_user_md(self) -> str:
        """Liest USER.md und gibt den bereinigten Inhalt zurück.
        Gibt leeren String zurück wenn die Datei nicht existiert oder
        nur Platzhalter-Text enthält.
        """
        if not self._user_md_pfad.exists():
            log.debug("USER.md nicht gefunden: %s", self._user_md_pfad)
            return ""

        try:
            raw = self._user_md_pfad.read_text(encoding="utf-8").strip()
        except OSError as e:
            log.warning("USER.md konnte nicht gelesen werden: %s", e)
            return ""

        if not raw:
            return ""

        # Kommentare und Überschriften (Struktur-Info) behalten,
        # Platzhalter-Zeilen wegfiltern
        zeilen = []
        for zeile in raw.splitlines():
            z = zeile.strip()
            if z.startswith("#"):
                zeilen.append(zeile)
                continue
            if not z:
                zeilen.append("")
                continue
            if any(marker in z for marker in _USER_MD_LEER_MARKER):
                continue
            zeilen.append(zeile)

        inhalt = "\n".join(zeilen).strip()

        # Nur zurückgeben wenn es echten Inhalt gibt (nicht nur Überschriften)
        hat_inhalt = any(
            z.strip() and not z.strip().startswith("#")
            for z in inhalt.splitlines()
        )
        if not hat_inhalt:
            return ""

        log.debug("USER.md geladen: %d Zeichen", len(inhalt))
        return inhalt

    @staticmethod
    def _aktuelles_datum_text() -> str:
        """Gibt das aktuelle Datum + Wochentag als Text zurück.
        Wird bei JEDEM system_prompt()-Aufruf neu berechnet.
        """
        jetzt = datetime.now()
        wochentag = _WOCHENTAGE[jetzt.weekday()]
        datum = jetzt.strftime("%d.%m.%Y")
        uhrzeit = jetzt.strftime("%H:%M")
        return f"{wochentag}, der {datum}, {uhrzeit} Uhr"

    def system_prompt(self, brain_fakten: list[str] | None = None) -> str:
        """Erzeugt den System-Prompt-Anteil für den LLM-Aufruf."""
        name = self._daten["name"]
        ton = self._daten["ton"]
        sprache = self._daten["sprache"]
        verboten = self._daten.get("verboten", [])
        verhalten = self._daten.get("verhalten", [])

        teile = [
            f"Du bist {name}, ein lokal auf der Hardware des Users laufender KI-Assistent.",
            f"Ton: {ton}. Sprache: {sprache}.",
        ]

        # Datum + Uhrzeit — bei jedem Aufruf aktuell
        teile.append(f"\nAktuelles Datum: {self._aktuelles_datum_text()}")
        teile.append(
            "Dieses Datum ist echt — es ist JETZT. Behandle es nicht als Zukunft "
            "und widersprich ihm nicht. Dein Trainings-Cutoff liegt in der Vergangenheit."
        )

        # Tool-Nutzung Instruktion
        teile.append(
            "\nWenn du nach aktuellen Ereignissen, News, Wetter, Börsenkursen oder "
            "Dingen gefragt wirst die du nicht sicher weißt: nutze die verfügbaren "
            "Tools (websearch, news, wetter, finanzen) BEVOR du 'weiß ich nicht' sagst. "
            "Weigere dich nie Informationen zu holen nur weil ein Datum dir unbekannt "
            "vorkommt — prüfe es mit den Tools."
        )

        # Datei-Output Format Instruktion
        teile.append(
            "\nWenn du Code, Konfigurationen oder Datei-Inhalte ausgibst die der User "
            "speichern oder verwenden soll: nutze dieses Format:\n"
            "DATEI: dateiname.ext\n---\n[Inhalt]\n---\n"
            "Das erzeugt eine herunterladbare Datei-Karte im Chat. "
            "Für kurze Inline-Code-Snippets die nur erklärt werden: normaler Code-Block ist OK."
        )

        if verhalten:
            teile.append("\nVerhalten:")
            for v in verhalten:
                teile.append(f"- {v}")

        if verboten:
            teile.append(f"Vermeide diese Phrasen: {', '.join(verboten)}")

        # USER.md — persistente User-Fakten (nur wenn befüllt)
        if self._user_md_inhalt:
            teile.append("\n## Persistente User-Fakten")
            teile.append(self._user_md_inhalt)

        # Brain-Fakten (dynamisch aus aktueller Suche)
        if brain_fakten:
            teile.append("\n## Bekannte Fakten aus dem Brain (aktiv nutzen):")
            for f_text in brain_fakten[:5]:
                teile.append(f"• {f_text}")

        return "\n".join(teile)

    @property
    def name(self) -> str:
        return self._daten.get("name", "Nova")

    @property
    def hat_user_fakten(self) -> bool:
        """True wenn USER.md echten Inhalt hat."""
        return bool(self._user_md_inhalt)

    def neu_laden(self) -> None:
        """Hot-Reload: persona.yaml + USER.md neu einlesen ohne Neustart."""
        self._daten = dict(_STANDARD)
        self._laden()
        log.info(
            "Persona neu geladen — USER.md: %s",
            "vorhanden" if self._user_md_inhalt else "leer",
        )
