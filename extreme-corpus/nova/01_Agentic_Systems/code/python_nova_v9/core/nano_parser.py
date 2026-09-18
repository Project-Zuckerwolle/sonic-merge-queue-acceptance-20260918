"""Nova v8 – NanoParser.

Leichtgewichtig: keine ML, nur Textanalyse.
Extrahiert Keywords, erkennt Todo-Signale und Intent-Hints.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from core.logger import get

log = get("parser")

STOPWOERTER = {
    "ich", "du", "er", "sie", "es", "wir", "ihr", "die", "der", "das",
    "ein", "eine", "einen", "einem", "einer", "eines",
    "ist", "bin", "bist", "sind", "war", "hat", "haben", "hatte",
    "und", "oder", "aber", "wenn", "dann", "dass", "damit",
    "auf", "in", "an", "bei", "mit", "von", "zu", "zum", "zur",
    "für", "aus", "nach", "über", "unter", "vor", "hinter",
    "nicht", "kein", "keine", "keinen", "keinem", "keiner",
    "wie", "was", "wer", "wo", "wann", "warum", "welche", "welcher",
    "mal", "noch", "schon", "auch", "nur", "sehr", "mehr",
    "kann", "muss", "will", "soll", "darf", "möchte",
    "bitte", "danke", "ok", "okay", "ja", "nein", "habe", "ihr",
    "mir", "mich", "dir", "dich", "ihm", "ihn", "uns",
    "this", "the", "a", "an", "is", "are", "was", "have", "has",
}

TODO_SIGNALE = [
    r"\berinnere\s+mich\b", r"\bnicht\s+vergessen\b", r"\baufgabe\b",
    r"\btodo\b", r"\bto[-\s]?do\b", r"\baufgaben\b", r"\bmerke?\s+mir\b",
    r"\breminder\b", r"\bwichtig\b.*\bmorgen\b", r"\bmorgen\b.*\bwichtig\b",
    r"\bmuss\s+ich\b", r"\bsollte\s+ich\b", r"\bnicht\s+vergessen\b",
]

# Intent-Hints für den Router/Orchestrator
INTENT_MUSTER = {
    "code":       [r"\bcode\b", r"\bpython\b", r"\bscript\b", r"\bprogramm\b", r"\bfunktion\b", r"\bfehler\b.*\bcode\b"],
    "suche":      [r"\bsuche?\b", r"\bfinde?\b", r"\bgoogle\b", r"\bweb\b", r"\bonline\b"],
    "wetter":     [r"\bwetter\b", r"\btemperatur\b", r"\bregen\b", r"\bsonne\b"],
    "datei":      [r"\bdatei\b", r"\border\b", r"\bverzeichnis\b", r"\blies\b", r"\bzeige?\b.*\bdatei\b"],
    "github":     [r"\bgit\b", r"\bgithub\b", r"\bcommit\b", r"\bpush\b", r"\brepo\b"],
    "research":   [r"\brecherchiere?\b", r"\banalyse?\b", r"\buntersuche?\b", r"\bberichte?\b", r"\bfasse?\s+zusammen\b"],
    "kalender":   [r"\btermin\b", r"\bmeeting\b", r"\bkalender\b", r"\bwann\b", r"\bum\s+\d+\s*uhr\b"],
    "erinnerung": [r"\berinner\b", r"\bvergiss?\s+nicht\b", r"\bnotiz\b"],
}


@dataclass
class ParseErgebnis:
    keywords:     list[str] = field(default_factory=list)
    todo_signal:  bool = False
    todo_text:    str = ""
    intents:      list[str] = field(default_factory=list)
    fragewort:    bool = False
    laenge:       int = 0


class NanoParser:
    def parse(self, text: str) -> ParseErgebnis:
        text_lower = text.lower()
        ergebnis = ParseErgebnis(laenge=len(text.split()))

        # Keywords extrahieren
        woerter = re.findall(r'\b[a-zäöüA-ZÄÖÜ][a-zäöüA-ZÄÖÜ]{2,}\b', text)
        ergebnis.keywords = list(dict.fromkeys(
            w.lower() for w in woerter if w.lower() not in STOPWOERTER
        ))[:12]

        # Todo-Signal
        for muster in TODO_SIGNALE:
            if re.search(muster, text_lower):
                ergebnis.todo_signal = True
                ergebnis.todo_text = text.strip()
                break

        # Intents
        for intent, muster_liste in INTENT_MUSTER.items():
            for muster in muster_liste:
                if re.search(muster, text_lower):
                    ergebnis.intents.append(intent)
                    break

        # Fragewort
        ergebnis.fragewort = bool(re.search(
            r'\b(was|wie|wo|wann|warum|wer|welche|welcher|kann|darf|soll|what|how|why|when|where|who)\b',
            text_lower
        ))

        log.debug(
            f"Parse | keywords={ergebnis.keywords} | intents={ergebnis.intents} | "
            f"todo={ergebnis.todo_signal} | frage={ergebnis.fragewort} | "
            f"laenge={ergebnis.laenge}"
        )
        return ergebnis

    def komprimiere(self, text: str, max_woerter: int = 50) -> str:
        """Kürzt Text auf max_woerter Wörter."""
        woerter = text.split()
        if len(woerter) <= max_woerter:
            return text
        return " ".join(woerter[:max_woerter]) + "…"
