"""Nova Predator v2 — NanoParser.
Regelbasiertes Keyword/Intent/Todo Parsing. Kein LLM, kein Nova-Import außer logger.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from core.logger import get

log = get("nano_parser")

_TODO_MUSTER = re.compile(
    r"\b(todo|aufgabe|erinnere mich|ich muss|ich soll|nicht vergessen|reminder)\b",
    re.IGNORECASE,
)
_PRAEFERENZ_MUSTER = re.compile(
    r"\b(ich mag|ich liebe|ich benutze|ich verwende|ich bevorzuge|ich hasse|ich arbeite mit)\b",
    re.IGNORECASE,
)
_FAKT_MUSTER = re.compile(
    r"\b(ist|sind|war|waren|heißt|bedeutet|funktioniert)\b",
    re.IGNORECASE,
)

# ── Plan-Trigger (v2) — öffnet Plan-Popup ────────────────────────────────────
_PLAN_TRIGGER: list[re.Pattern] = [
    re.compile(r"^/plan\b", re.IGNORECASE),
    re.compile(r"\b(planungs|planungs-?)phase\b", re.IGNORECASE),
    re.compile(r"\blass uns (das |ein |einen |dieses )?(projekt |spiel |system |feature )?plan(en|nen)\b", re.IGNORECASE),
    re.compile(r"\bich möchte (das |ein |einen )?(projekt )?plan(en|nen)\b", re.IGNORECASE),
    re.compile(r"\b(können|wollen) wir (das |das Projekt |das Spiel )?(planen|einen Plan)\b", re.IGNORECASE),
    re.compile(r"\bmach(e|en)? (einen )?plan\b", re.IGNORECASE),
]

# ── Skill-Routing Tier-2 hints (für LLM-Selector-Fallback) ──────────────────
_TOOL_HINTS: dict[str, list[str]] = {
    "websearch":  ["suche", "google", "recherch", "finde", "was ist", "aktuell", "news"],
    "wetter":     ["wetter", "temperatur", "regen", "sonne", "wind", "grad"],
    "news":       ["news", "nachrichten", "aktuell", "heute", "schlagzeilen"],
    "finanzen":   ["aktie", "kurs", "bitcoin", "crypto", "börse", "preis"],
    "brain":      ["erinnere", "weißt du", "ich hatte", "du hast gesagt"],
    "datei":      ["datei", "ordner", "speichere", "öffne", "lese"],
    "bash":       ["führe aus", "terminal", "befehl", "shell", "script starten"],
}

_STOPWOERTER = frozenset({
    "ich", "du", "er", "sie", "es", "wir", "ihr", "die", "der", "das",
    "ein", "eine", "und", "oder", "aber", "für", "mit", "von", "bei",
    "im", "ist", "sind", "war", "hat", "haben", "dass", "wie", "was",
    "wer", "wo", "wann", "nicht", "auch", "noch", "schon", "ja", "nein",
})

_INTENT_PATTERN: list[tuple[str, re.Pattern]] = [
    ("wetter", re.compile(r"\b(wetter|temperatur|regen|sonne|wind)\b", re.IGNORECASE)),
    ("code", re.compile(r"\b(code|python|script|funktion|klasse|bug|fehler|programmier)\b", re.IGNORECASE)),
    ("suche", re.compile(r"\b(suche?|finde?|google|recherch|such nach)\b", re.IGNORECASE)),
    ("kalender", re.compile(r"\b(termin|kalender|meeting|treffen|datum|uhrzeit|wann)\b", re.IGNORECASE)),
    ("datei", re.compile(r"\b(datei|ordner|verzeichnis|speichere?|lese?|öffne?)\b", re.IGNORECASE)),
    ("todo", re.compile(r"\b(todo|aufgabe|erledigen|abgehakt)\b", re.IGNORECASE)),
    ("brain", re.compile(r"\b(erinnere dich|weißt du noch|ich hatte|du hast gesagt)\b", re.IGNORECASE)),
]


@dataclass
class ParseErgebnis:
    keywords: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    todo_signal: bool = False
    todo_text: str = ""
    praeferenz_signal: bool = False
    fakt_signal: bool = False
    # v2-Erweiterungen
    plan_trigger: bool = False
    tool_hints: list[str] = field(default_factory=list)
    # v3.3: Thinking-Mode Signal
    komplex: bool = False   # True → qwen3:8b mit think:true verwenden


class NanoParser:
    def parse(self, text: str) -> ParseErgebnis:
        ergebnis = ParseErgebnis()

        # Keywords: Wörter > 3 Zeichen, keine Stopwörter
        woerter = re.findall(r"\b\w{3,}\b", text.lower())
        ergebnis.keywords = list({w for w in woerter if w not in _STOPWOERTER})[:20]

        # Intents
        for name, muster in _INTENT_PATTERN:
            if muster.search(text):
                ergebnis.intents.append(name)

        # Todo-Erkennung
        if _TODO_MUSTER.search(text):
            ergebnis.todo_signal = True
            # Todo-Text: alles nach dem Trigger-Wort
            m = _TODO_MUSTER.search(text)
            if m:
                ergebnis.todo_text = text[m.end():].strip().rstrip(".")

        # Präferenz-Erkennung
        if _PRAEFERENZ_MUSTER.search(text):
            ergebnis.praeferenz_signal = True

        # Fakt-Erkennung (grob)
        if _FAKT_MUSTER.search(text) and len(text) > 20:
            ergebnis.fakt_signal = True

        # Plan-Trigger (v2): öffnet Plan-Popup
        for muster in _PLAN_TRIGGER:
            if muster.search(text):
                ergebnis.plan_trigger = True
                break

        # Tool-Hints (v2): Ebene-1 Skill-Routing
        text_lower = text.lower()
        for tool, keywords in _TOOL_HINTS.items():
            if any(kw in text_lower for kw in keywords):
                ergebnis.tool_hints.append(tool)

        # v3.3: Komplexitäts-Erkennung → Thinking-Mode
        # LLM selbst entscheidet via /think token — wir geben nur den Hinweis
        ergebnis.komplex = self._ist_komplex(text)

        return ergebnis

    @staticmethod
    def _ist_komplex(text: str) -> bool:
        """Erkennt ob eine Frage Reasoning braucht.

        Kriterien (eines reicht):
          - Analyse/Vergleich/Planung Keywords
          - Mehrere Teilfragen (? mehr als einmal, oder 'und ... und')
          - Länge > 120 Zeichen (detaillierte Anfragen)
          - Begründungs-Fragen (warum, weshalb, wieso)
          - Code-Architektur, Konzepte, Erklärungen
        """
        t = text.lower()

        # Begründungs- und Analyse-Keywords
        komplex_keywords = (
            "warum", "weshalb", "wieso", "erkläre", "erkläre mir",
            "analysiere", "vergleiche", "unterschied", "vor- und nachteile",
            "wie funktioniert", "wie kann ich", "was ist besser",
            "optimiere", "verbessere", "refaktor", "architektur",
            "konzept", "strategie", "plan", "entscheide", "empfehle",
            "sollte ich", "was wäre", "was würde", "wie würde",
            "schritt für schritt", "erkläre ausführlich",
        )
        if any(kw in t for kw in komplex_keywords):
            return True

        # Mehrere Teilfragen
        if text.count("?") >= 2:
            return True

        # Lange detaillierte Anfrage
        if len(text) > 120:
            return True

        return False
