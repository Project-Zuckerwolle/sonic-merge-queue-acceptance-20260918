"""Nova Predator v1 — BrainGate.

Das Qualitätstor für Brain-Writes.
Entscheidet ob ein extrahierter Fakt es wert ist, dauerhaft im Brain gespeichert
zu werden — und wenn ja, mit welchem Gewicht.

Dreistufige Prüfung:
  1. Schnellfilter (kein LLM, microsekunden):
       - Länge unter 15 Zeichen → ablehnen
       - Blacklist-Match → ablehnen
       - Turn-Limit erreicht → ablehnen

  2. LLM-Bewertung (qwen2.5:3b, ~0.3–0.5s):
       Bewertet 4 Dimensionen je 0–10:
       - Persistenz:    Wird das in 30 Tagen noch relevant sein?
       - Spezifitaet:   Ist es konkret und spezifisch?
       - Personalitaet: Bezieht es sich auf den User oder sein System?
       - Neuheit:       Ist es verglichen mit bekannten Einträgen neu?

       Gewichteter Score = Persistenz*0.40 + Spezifitaet*0.30
                         + Personalitaet*0.20 + Neuheit*0.10

  3. Entscheidung:
       score < MIN_SCORE  → abgelehnt (kein Brain-Write)
       score < NOTIZ_GRENZE → "notiz" (kein Embedding)
       score >= NOTIZ_GRENZE → "vollstaendig" (mit Embedding)

Konfiguration via config.yaml (brain.gate.*).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.logger import get

if TYPE_CHECKING:
    from core.ollama_client import OllamaClient

log = get("brain_gate")

# ── Blacklist — diese Muster sofort ablehnen (keine LLM-Zeit verschwenden) ──
# WICHTIG: Einträge ohne führende/nachfolgende Spaces — wird mit Wortgrenzen geprüft.

_BLACKLIST: frozenset[str] = frozenset({
    # Begrüßungen
    "guten morgen", "guten abend", "guten tag", "guten nachmittag",
    "hallo", "hi", "hey",
    "tschüss", "tschüs", "auf wiedersehen", "bis dann", "bis bald",
    "ciao",
    # Kurzbestätigungen — nur als Standalone sinnlos
    "ok", "okay",
    "alles klar", "verstanden",
    "kein problem",
    # Nova-Feedback (macht keinen Sinn als Langzeit-Wissen)
    "nova ist toll", "nova ist gut", "nova ist super",
    "ich finde dich", "das war hilfreich",
    # Inhaltsleere Einleitungen
    "wie geht es", "was ist los",
})

# ── LLM-Prompt ───────────────────────────────────────────────────────────────

_GATE_PROMPT = """\
Du entscheidest ob ein Fakt dauerhaft in ein persönliches KI-Gedächtnis gespeichert werden soll.

Fakt: "{fakt}"
Quelle: {quelle}
Bereits bekannte ähnliche Einträge (zur Einschätzung der Neuheit):
{bekannt}

Bewerte auf einer Skala 0–10:
- persistenz: Wird das in 30 Tagen noch relevant sein? (Smalltalk=0, Projektwissen=9)
- spezifitaet: Ist es konkret und präzise? ("ich arbeite"=1, "nutze Python 3.14.4 auf Windows"=9)
- personalitaet: Bezieht es sich auf den User oder sein System? (Allgemeinwissen=1, persönlich=9)
- neuheit: Wie neu ist das verglichen mit den bekannten Einträgen? (identisch=0, völlig neu=10)

Antworte NUR mit JSON (kein anderer Text):
{{"persistenz": N, "spezifitaet": N, "personalitaet": N, "neuheit": N, "ablehnungsgrund": ""}}

Wenn Score unter 5: Begründe kurz im ablehnungsgrund-Feld warum es nicht würdig ist.\
"""

# ── Ergebnis-Dataclass ────────────────────────────────────────────────────────

@dataclass
class GateResult:
    """Ergebnis einer BrainGate-Bewertung."""
    score: float          # Gewichteter Score 0.0–10.0
    akzeptiert: bool      # True = Brain-Write erlaubt
    gewicht: str          # "vollstaendig" | "notiz" | "abgelehnt"
    ablehnungsgrund: str  # Begründung wenn abgelehnt (für Logging)

    @property
    def mit_embedding(self) -> bool:
        """True wenn dieser Entry ein Embedding bekommen soll."""
        return self.gewicht == "vollstaendig"


# ── Gate-Klasse ───────────────────────────────────────────────────────────────

class BrainGate:
    """LLM-gestützter Qualitätsfilter für Brain-Writes.

    Wird von BrainExtractor vor brain_manager.add() aufgerufen.
    Eine Instanz pro BrainExtractor — turn_reset() nach jedem Chat-Turn.
    """

    def __init__(
        self,
        ollama: "OllamaClient",
        modell: str = "qwen2.5:3b",
        min_score: float = 5.0,
        notiz_grenze: float = 7.0,
        max_pro_turn: int = 3,
    ) -> None:
        self._ollama = ollama
        self._modell = modell
        self._min_score = min_score
        self._notiz_grenze = notiz_grenze
        self._max_pro_turn = max_pro_turn
        self._turn_count = 0

    def turn_reset(self) -> None:
        """Muss nach jedem abgeschlossenen Chat-Turn aufgerufen werden.
        Setzt den Turn-Zähler zurück.
        """
        self._turn_count = 0

    async def bewerte(
        self,
        fakt: str,
        quelle: str = "chat",
        bekannte_inhalte: list[str] | None = None,
    ) -> GateResult:
        """Bewertet ob ein Fakt ins Brain geschrieben werden soll.

        Args:
            fakt:             Der zu bewertende Fakt-Text.
            quelle:           Herkunft ('chat', 'apex', 'manuell').
            bekannte_inhalte: Texte ähnlicher bestehender Brain-Entries
                              (für Neuheits-Einschätzung).

        Returns:
            GateResult mit score, akzeptiert, gewicht, ablehnungsgrund.
        """
        fakt = fakt.strip()

        # ── Schnellfilter 1: Länge ───────────────────────────────────
        if len(fakt) < 15:
            log.debug("Gate ablehnt (zu kurz %d): '%s'", len(fakt), fakt[:40])
            return GateResult(0.0, False, "abgelehnt", "zu kurz")

        # ── Schnellfilter 2: Blacklist ───────────────────────────────
        fakt_lower = fakt.lower()
        for phrase in _BLACKLIST:
            # Wortgrenz-Check: Phrase muss als vollständiges Wort/Phrase vorkommen
            # Verhindert falsche Treffer wie " ja" in "javascript"
            import re as _re
            phrase_escaped = _re.escape(phrase.strip())
            if _re.search(r'(?<!\w)' + phrase_escaped + r'(?!\w)', fakt_lower):
                log.debug("Gate ablehnt (blacklist '%s'): '%s'", phrase.strip(), fakt[:40])
                return GateResult(0.0, False, "abgelehnt", f"blacklist: {phrase.strip()}")

        # ── Schnellfilter 3: Turn-Limit ──────────────────────────────
        if self._turn_count >= self._max_pro_turn:
            log.debug("Gate ablehnt (turn-limit %d): '%s'", self._max_pro_turn, fakt[:40])
            return GateResult(0.0, False, "abgelehnt", f"turn-limit {self._max_pro_turn} erreicht")

        # ── LLM-Bewertung ─────────────────────────────────────────────
        bekannt = bekannte_inhalte[:5] if bekannte_inhalte else []
        bekannt_text = (
            "\n".join(f"  - {t[:100]}" for t in bekannt)
            if bekannt else "  (Brain ist leer oder keine ähnlichen Einträge gefunden)"
        )

        prompt = _GATE_PROMPT.format(
            fakt=fakt[:250],
            quelle=quelle,
            bekannt=bekannt_text,
        )

        try:
            antwort = await self._ollama.chat(
                nachrichten=[{"role": "user", "content": prompt}],
                modell=self._modell,
                optionen={"temperature": 0.1, "num_predict": 200},
            )
            daten = self._parse_json(antwort)
        except Exception as e:
            # Bei LLM-Fehler: Fallback-Akzeptanz mit mittlerem Score
            # Lieber zu viel als gar nichts wenn das Brain-LLM nicht erreichbar ist
            log.warning("Gate LLM-Fehler (Fallback-Akzeptanz): %s", e)
            self._turn_count += 1
            return GateResult(6.0, True, "notiz", "")

        if daten is None:
            log.warning("Gate: LLM-Antwort nicht parsebar, Fallback-Akzeptanz")
            self._turn_count += 1
            return GateResult(6.0, True, "notiz", "")

        # Score berechnen
        score = (
            float(daten.get("persistenz",    5)) * 0.40 +
            float(daten.get("spezifitaet",   5)) * 0.30 +
            float(daten.get("personalitaet", 5)) * 0.20 +
            float(daten.get("neuheit",       5)) * 0.10
        )
        ablehnungsgrund = str(daten.get("ablehnungsgrund", ""))

        if score < self._min_score:
            log.debug("Gate ablehnt (score=%.1f): '%s' — %s",
                      score, fakt[:40], ablehnungsgrund)
            return GateResult(score, False, "abgelehnt", ablehnungsgrund or "score zu niedrig")

        # Akzeptiert
        self._turn_count += 1
        gewicht = "vollstaendig" if score >= self._notiz_grenze else "notiz"
        log.debug("Gate akzeptiert (score=%.1f, %s): '%s'", score, gewicht, fakt[:40])
        return GateResult(score, True, gewicht, "")

    def _parse_json(self, text: str) -> dict | None:
        """Extrahiert JSON aus LLM-Antwort tolerant gegenüber Markdown-Fences."""
        import re
        # Markdown-Fences entfernen
        bereinigt = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
        start = bereinigt.find("{")
        end = bereinigt.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            return json.loads(bereinigt[start:end + 1])
        except json.JSONDecodeError:
            return None

    @property
    def turn_writes(self) -> int:
        """Anzahl akzeptierter Writes im aktuellen Turn."""
        return self._turn_count
