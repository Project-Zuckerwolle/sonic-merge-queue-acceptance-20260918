"""Nova Predator v2 — FuzzyDedup.
Schnelle Ähnlichkeitsprüfung für Brain-Entries via Levenshtein-Ratio.
Verhindert Duplikate durch Tippfehler, Groß/Klein-Varianten und
minimale Reformulierungen.

Kein LLM, kein Embedding — rein string-basiert.
Ergänzt den bestehenden Vektor-Duplikat-Check im BrainExtractor.

Threshold 85 (0-100 Skala):
  "Python 3.14 wird genutzt" vs "Python 3.14 wird verwendet" → ~87 → DUPLIKAT
  "Python 3.14 wird genutzt" vs "Java wird genutzt" → ~55 → KEIN DUPLIKAT
"""
from __future__ import annotations
from core.logger import get

log = get("fuzzy_dedup")

# Threshold auf 0-100 Skala (rapidfuzz-Stil)
STANDARD_THRESHOLD = 85


def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Berechnet Levenshtein-Ratio ohne externe Bibliotheken.
    Gibt Wert 0-100 zurück (wie rapidfuzz).
    Fallback wenn rapidfuzz nicht installiert.
    """
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if s1 == s2:
        return 100.0
    if not s1 or not s2:
        return 0.0

    m, n = len(s1), len(s2)
    # Vereinfachte Edit-Distance
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    edit_dist = dp[n]
    max_len = max(m, n)
    return round((1 - edit_dist / max_len) * 100, 1)


try:
    from rapidfuzz import fuzz as _rfuzz
    _HAS_RAPIDFUZZ = True
    log.debug("rapidfuzz verfügbar — nutze token_sort_ratio")
except ImportError:
    _HAS_RAPIDFUZZ = False
    log.debug("rapidfuzz nicht installiert — nutze Levenshtein-Fallback")


def aehnlichkeit(text1: str, text2: str) -> float:
    """Gibt Ähnlichkeit 0-100 zurück.
    Nutzt rapidfuzz wenn verfügbar, sonst Levenshtein-Fallback.
    token_sort_ratio ignoriert Wort-Reihenfolge.
    """
    if not text1 or not text2:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return _rfuzz.token_sort_ratio(text1, text2)
    return _levenshtein_ratio(text1, text2)


def ist_duplikat(
    neu: str,
    vorhandene: list[str],
    threshold: int = STANDARD_THRESHOLD,
) -> tuple[bool, str, float]:
    """Prüft ob 'neu' einem der vorhandenen Texte ähnlich ist.

    Returns:
        (ist_duplikat, bestes_match, score)
        ist_duplikat: True wenn score >= threshold
        bestes_match: der ähnlichste vorhandene Text
        score: Ähnlichkeit 0-100
    """
    if not vorhandene:
        return False, "", 0.0

    bestes_match = ""
    bester_score = 0.0

    for v in vorhandene:
        score = aehnlichkeit(neu, v)
        if score > bester_score:
            bester_score = score
            bestes_match = v

    return bester_score >= threshold, bestes_match, bester_score
