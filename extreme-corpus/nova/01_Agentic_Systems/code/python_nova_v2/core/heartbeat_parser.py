"""Nova Predator v2 — HeartbeatParser.
Parst HEARTBEAT.md mit natürlichsprachlichen Schedules und wandelt sie
in Scheduler-Jobs um.

Format in HEARTBEAT.md:
  Jeden Morgen um 07:00 Uhr erstelle ein Briefing...
  Jeden Montag um 09:00 Uhr...
  Alle 30 Minuten...
  Täglich um 03:00 Uhr...

Zeilen die mit # beginnen werden ignoriert.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path

from core.logger import get

log = get("heartbeat_parser")

_HEARTBEAT_PFAD = Path("behavior/HEARTBEAT.md")

# Muster für Zeit-Extraktion
_UHRZEIT_MUSTER = re.compile(r"\b(\d{1,2}):(\d{2})\s*(?:Uhr)?\b")
_INTERVALL_MUSTER = re.compile(
    r"\balle?\s+(\d+)\s*(minute[n]?|stunde[n]?|sekunde[n]?)\b",
    re.IGNORECASE,
)
_TAEGLICH_MUSTER = re.compile(r"\b(täglich|jeden tag|daily)\b", re.IGNORECASE)
_WOECHENTLICH_MUSTER = re.compile(
    r"\b(jeden\s+)?(montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag)\b",
    re.IGNORECASE,
)

_WOCHENTAG_MAP = {
    "montag": 0, "dienstag": 1, "mittwoch": 2, "donnerstag": 3,
    "freitag": 4, "samstag": 5, "sonntag": 6,
}


@dataclass
class ScheduleEntry:
    """Ein geparster Schedule-Eintrag."""
    beschreibung: str          # Original-Text der Aufgabe
    typ: str                   # "uhrzeit" | "intervall" | "taeglich"
    uhrzeit: str | None = None # "07:00" für tägliche Jobs
    intervall_s: int | None = None  # Sekunden für Intervall-Jobs
    wochentag: int | None = None    # 0=Montag, 6=Sonntag


class HeartbeatParser:
    """Parst HEARTBEAT.md und extrahiert Schedule-Einträge."""

    def __init__(self, pfad: Path = _HEARTBEAT_PFAD) -> None:
        self._pfad = pfad

    def laden_und_parsen(self) -> list[ScheduleEntry]:
        """Lädt und parst HEARTBEAT.md."""
        if not self._pfad.exists():
            log.debug("HEARTBEAT.md nicht gefunden: %s", self._pfad)
            return []
        try:
            inhalt = self._pfad.read_text(encoding="utf-8")
            return self.parsen(inhalt)
        except Exception as e:
            log.warning("HEARTBEAT.md Fehler: %s", e)
            return []

    def parsen(self, text: str) -> list[ScheduleEntry]:
        """Parst HEARTBEAT-Text in Schedule-Einträge."""
        entries = []
        aktuell_beschreibung: list[str] = []
        aktuell_trigger: str | None = None

        for zeile in text.splitlines():
            zeile_stripped = zeile.strip()

            # Kommentare und Leerzeilen
            if not zeile_stripped or zeile_stripped.startswith("#"):
                # Wenn wir eine akkumulierte Beschreibung haben, abschließen
                if aktuell_trigger and aktuell_beschreibung:
                    entry = self._erstelle_entry(
                        aktuell_trigger,
                        " ".join(aktuell_beschreibung)
                    )
                    if entry:
                        entries.append(entry)
                aktuell_beschreibung = []
                aktuell_trigger = None
                continue

            # Neue Zeile: prüfe ob sie einen Trigger enthält
            hat_trigger = self._hat_trigger(zeile_stripped)

            if hat_trigger:
                # Vorherige Beschreibung abschließen
                if aktuell_trigger and aktuell_beschreibung:
                    entry = self._erstelle_entry(
                        aktuell_trigger,
                        " ".join(aktuell_beschreibung)
                    )
                    if entry:
                        entries.append(entry)
                aktuell_trigger = zeile_stripped
                aktuell_beschreibung = [zeile_stripped]
            elif aktuell_trigger:
                # Fortsetzung der aktuellen Beschreibung
                aktuell_beschreibung.append(zeile_stripped)

        # Letzten Entry abschließen
        if aktuell_trigger and aktuell_beschreibung:
            entry = self._erstelle_entry(
                aktuell_trigger,
                " ".join(aktuell_beschreibung)
            )
            if entry:
                entries.append(entry)

        log.info("HEARTBEAT.md: %d Schedules geladen", len(entries))
        return entries

    def _hat_trigger(self, text: str) -> bool:
        """Prüft ob eine Zeile einen Zeitauslöser enthält."""
        text_lower = text.lower()
        return bool(
            _UHRZEIT_MUSTER.search(text) or
            _INTERVALL_MUSTER.search(text) or
            _TAEGLICH_MUSTER.search(text) or
            _WOECHENTLICH_MUSTER.search(text)
        )

    def _erstelle_entry(self, trigger: str, beschreibung: str) -> ScheduleEntry | None:
        """Erstellt einen ScheduleEntry aus Trigger-Text."""
        # Intervall-Check
        m = _INTERVALL_MUSTER.search(trigger)
        if m:
            zahl = int(m.group(1))
            einheit = m.group(2).lower()
            if "minute" in einheit:
                sekunden = zahl * 60
            elif "stunde" in einheit:
                sekunden = zahl * 3600
            else:
                sekunden = zahl
            return ScheduleEntry(
                beschreibung=beschreibung,
                typ="intervall",
                intervall_s=sekunden,
            )

        # Uhrzeit-Check
        m_zeit = _UHRZEIT_MUSTER.search(trigger)
        uhrzeit = f"{int(m_zeit.group(1)):02d}:{m_zeit.group(2)}" if m_zeit else None

        # Wochentag-Check
        m_wt = _WOECHENTLICH_MUSTER.search(trigger)
        if m_wt and uhrzeit:
            tag_name = m_wt.group(2).lower() if m_wt.group(2) else ""
            wochentag = _WOCHENTAG_MAP.get(tag_name)
            return ScheduleEntry(
                beschreibung=beschreibung,
                typ="woechentlich",
                uhrzeit=uhrzeit,
                wochentag=wochentag,
            )

        # Täglich
        if uhrzeit:
            return ScheduleEntry(
                beschreibung=beschreibung,
                typ="taeglich",
                uhrzeit=uhrzeit,
            )

        return None
