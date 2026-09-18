"""Kalender-Skill: Termine erkennen, ins Brain schreiben und abrufen."""
from __future__ import annotations
import re
from datetime import datetime
from core.skill_registry import SkillContext, SkillResult

# Muster für Datumsangaben
DATUM_MUSTER = [
    r'\b(\d{1,2})\.\s*(\d{1,2})\.?\s*(\d{4})?\b',           # 15.05 oder 15.05.2025
    r'\b(morgen|übermorgen|heute)\b',
    r'\b(montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag)\b',
    r'\b(\d{1,2})\s*(?:uhr|:)\s*(\d{0,2})\b',               # 14 Uhr oder 14:30
]

TERMIN_ERKENNUNGS_WOERTER = {
    "meeting", "termin", "arzt", "zahnarzt", "geburtstag", "feier",
    "konferenz", "call", "gespräch", "besprechung", "event", "veranstaltung",
    "erinnerung", "erinnere", "reminder", "trage ein", "notiere",
}

ABFRAGE_WOERTER = {
    "was steht", "zeig mir", "welche termine", "meine termine",
    "was habe ich", "nächste woche", "diese woche", "heute",
}


def on_message(ctx: SkillContext) -> SkillResult | None:
    text_lower = ctx.user_input.lower()

    # Ist es eine Abfrage oder ein neuer Termin?
    ist_abfrage = any(w in text_lower for w in ABFRAGE_WOERTER)
    ist_termin  = (
        any(w in text_lower for w in TERMIN_ERKENNUNGS_WOERTER) or
        any(re.search(m, text_lower) for m in DATUM_MUSTER[:2])
    )

    if ist_abfrage and not ist_termin:
        # Termine aus Brain-Hits laden
        return _zeige_termine(ctx)

    if ist_termin:
        # Termin-Details extrahieren und anbieten
        return _extrahiere_termin(ctx)

    return None


def _zeige_termine(ctx: SkillContext) -> SkillResult:
    # Termine aus Brain-Hits filtern
    termin_hits = [
        h for h in ctx.brain_hits
        if h.get("typ") == "termin" or "termin" in h.get("tags", [])
    ]
    if not termin_hits:
        return SkillResult(
            inhalt="Keine Termine im Brain gefunden. Sage mir z.B.: 'Meeting morgen 14 Uhr'.",
            typ="info"
        )
    zeilen = [f"• {h.get('titel','?')} ({h.get('erstellt','')[:10]})" for h in termin_hits[:8]]
    return SkillResult(
        inhalt=f"Deine Termine ({len(termin_hits)}):\n" + "\n".join(zeilen),
        typ="info"
    )


def _extrahiere_termin(ctx: SkillContext) -> SkillResult:
    text = ctx.user_input.strip()

    # Datum/Zeit-Info extrahieren
    datum_info = []
    for muster in DATUM_MUSTER:
        m = re.search(muster, text.lower())
        if m:
            datum_info.append(m.group())

    # Titel aus Keywords
    termin_woerter = [kw for kw in ctx.keywords
                      if kw not in {"heute", "morgen", "uhr", "am", "um"}]
    titel = " ".join(termin_woerter[:4]).title() if termin_woerter else text[:40]

    zeitangabe = ", ".join(datum_info) if datum_info else "Zeitangabe unklar"

    return SkillResult(
        inhalt=(
            f"Termin erkannt: **{titel}**\n"
            f"Zeit: {zeitangabe}\n"
            f"Originaltext: {text[:150]}\n\n"
            f"*(Wird als Brain-Entry gespeichert)*"
        ),
        typ="aktion",
        metadaten={
            "termin_titel": titel,
            "zeitangabe": zeitangabe,
            "originaltext": text,
            # Signal an BrainWriter: als Termin speichern
            "brain_typ": "termin",
            "brain_tags": ["termin", "kalender"],
        }
    )
