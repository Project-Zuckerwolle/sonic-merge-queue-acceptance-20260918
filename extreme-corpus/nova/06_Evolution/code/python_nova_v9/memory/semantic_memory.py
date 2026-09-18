"""Nova v9 – Semantic Memory.

Langzeitgedaechtnis: Wrapper um den BrainManager.
Unveraendert von v8 uebernommen – war solide.
Bietet semantische Suche, Graph-Traversal, Tag-Filter.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from core.logger import get

if TYPE_CHECKING:
    from core.brain_manager import BrainManager
    from core.ollama_client import OllamaClient

log = get("semantic_mem")


class SemanticMemory:
    def __init__(self, brain: "BrainManager", ollama: "OllamaClient") -> None:
        self._brain  = brain
        self._ollama = ollama

    def suche(
        self,
        query: str,
        top_n: int = 5,
        tags:  list[str] | None = None,
    ) -> list[dict]:
        """Semantische Suche im Brain, optional mit Tag-Filter."""
        q_vec = self._ollama.embed(query)
        if q_vec is not None:
            treffer = self._brain.suche_semantisch(q_vec, top_n=top_n * 2)
            log.debug(f"Brain-Suche | '{query[:50]}' | Vektor | {len(treffer)} Kandidaten")
        else:
            treffer = self._brain.suche_fuzzy(query.split(), top_n=top_n * 2)
            log.debug(f"Brain-Suche | '{query[:50]}' | Fuzzy-Fallback | {len(treffer)} Kandidaten")

        if tags:
            treffer = [t for t in treffer if any(tag in t.get("tags", []) for tag in tags)]

        result = treffer[:top_n]
        log.debug(f"Brain-Ergebnis | {len(result)} Treffer | scores={[round(t.get('aehnlichkeit', 0), 2) for t in result]}")
        return result

    def nachbarn(self, entry_id: str, hops: int = 1) -> list[dict]:
        """Gibt Nachbarn eines Brain-Eintrags zurueck (Graph-Traversal)."""
        besucht: set[str] = {entry_id}
        aktuell: set[str] = {entry_id}
        alle: list[dict] = []

        for hop in range(hops):
            naechste: set[str] = set()
            for eid in aktuell:
                for conn in self._brain.connections_laden(eid):
                    ziel_id = conn.get("id", "")
                    if ziel_id and ziel_id not in besucht:
                        naechste.add(ziel_id)
                        besucht.add(ziel_id)
                        inhalt = self._brain.eintrag_laden(ziel_id)
                        if inhalt:
                            alle.append({
                                "id":       ziel_id,
                                "inhalt":   inhalt[:300],
                                "relation": conn.get("relation", ""),
                                "hop":      hop + 1,
                            })
            aktuell = naechste
            if not aktuell:
                break

        return alle

    def status(self) -> dict:
        stats = self._brain.get_stats()
        return {
            "brain_eintraege": stats.get("gesamt", 0),
            "mit_embedding":   stats.get("mit_embedding", 0),
        }
