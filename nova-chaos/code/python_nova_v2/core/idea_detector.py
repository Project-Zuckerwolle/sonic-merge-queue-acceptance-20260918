"""Nova Predator v1 — IdeaDetector.
Berechnet einen Durchbruch-Score für Brain-Entries vom Typ 'idee'.

Score-Dimensionen (0.0–1.0 gesamt):
  [0.30] Vernetzungsgrad    — wie viele andere Entries verlinkt?
  [0.25] Themen-Diversität  — wie viele verschiedene Tags der verlinkten Entries?
  [0.25] Neuheits-Index     — wie weit weg vom nächsten bekannten Eintrag?
  [0.20] Themen-Reife       — wie oft taucht das Thema im Brain auf?

Schwellenwerte:
  0.00–0.49  → normal
  0.50–0.74  → highlight
  0.75–0.89  → durchbruch  → EventBus: BREAKTHROUGH_IDEA
  0.90–1.00  → bedeutsam   → EventBus + aktiver UI-Push
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from core.logger import get

if TYPE_CHECKING:
    from core.brain_index import BrainIndex
    from core.brain_manager import BrainEntry, BrainManager

log = get("idea_detector")

SCHWELLE_HIGHLIGHT   = 0.50
SCHWELLE_DURCHBRUCH  = 0.75
SCHWELLE_BEDEUTSAM   = 0.90


def berechne_score(
    idee: "BrainEntry",
    alle_entries: list["BrainEntry"],
    brain_index: "BrainIndex",
) -> float:
    """Berechnet den Durchbruch-Score einer Idee. Gibt 0.0–1.0 zurück."""
    score = 0.0

    # ── Dimension 1: Vernetzungsgrad (max 0.30) ──────────────────────
    links_count = len(idee.links)
    score += min(links_count / 10.0, 0.30)

    # ── Dimension 2: Themen-Diversität der verlinkten Entries (max 0.25)
    if idee.links and alle_entries:
        id_zu_entry = {e.id: e for e in alle_entries}
        verlinkte_tags: set[str] = set()
        for link_id in idee.links:
            e = id_zu_entry.get(link_id)
            if e:
                verlinkte_tags.update(e.tags)
        score += min(len(verlinkte_tags) / 8.0, 0.25)

    # ── Dimension 3: Neuheits-Index (max 0.25) ───────────────────────
    # Niedrige Ähnlichkeit zum nächsten Nachbarn = neuartiger
    if idee.vektor:
        naechste_sim = brain_index.naechste_aehnlichkeit(idee.vektor)
        score += (1.0 - naechste_sim) * 0.25
    else:
        score += 0.125   # Kein Vektor → neutraler Wert

    # ── Dimension 4: Themen-Reife im Brain (max 0.20) ────────────────
    # Themen über die viel gesprochen wird → Brain hat Kontext
    if idee.tags and alle_entries:
        idee_tags = set(idee.tags)
        treffer = sum(
            1 for e in alle_entries
            if set(e.tags) & idee_tags
        )
        score += min(treffer / 20.0, 0.20)

    final = round(min(1.0, max(0.0, score)), 3)
    log.debug(
        "IdeaScore '%s': %.3f (links=%d)",
        idee.id[:8], final, links_count,
    )
    return final


def klassifiziere(score: float) -> str:
    """Gibt Klassifikation als String zurück."""
    if score >= SCHWELLE_BEDEUTSAM:
        return "bedeutsam"
    if score >= SCHWELLE_DURCHBRUCH:
        return "durchbruch"
    if score >= SCHWELLE_HIGHLIGHT:
        return "highlight"
    return "normal"


def ist_durchbruch(score: float) -> bool:
    return score >= SCHWELLE_DURCHBRUCH
