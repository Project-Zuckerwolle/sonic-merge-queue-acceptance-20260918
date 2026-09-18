"""Nova Predator v3.3 — SemanticMemory.
Sucht im BrainIndex nach relevanten Entries für den aktuellen Context.
Bridge zwischen Layer 1 (Memory) und Layer 0 (Brain).

Predator-Upgrade:
  - Nutzt HybridSearch (Vector + BM25 + Recency) statt reinem Cosinus-Search
  - Erhöht abruf_count für jeden gefundenen Entry (Feedback-Loop für Decay)
  - query_text Parameter für BM25-Keyword-Matching

v3.3:
  - als_fakten_text() holt 15 Kandidaten (statt 5)
  - Optionale LLM-Komprimierung: 15 Rohtreffer → 1 prägnanter Kontext-Absatz
  - Cache (5 Min TTL) verhindert Re-Generierung bei ähnlichen Fragen
"""
from __future__ import annotations
import asyncio
import hashlib
import time
from typing import TYPE_CHECKING

from core.logger import get

if TYPE_CHECKING:
    from core.brain_manager import BrainEntry, BrainManager
    from core.brain_index import BrainIndex, HybridSearch
    from core.ollama_client import OllamaClient

log = get("semantic_memory")

_KOMPRIMIER_PROMPT = """\
Diese Brain-Einträge sind aus einem KI-Assistenten-Gedächtnis:

{fakten}

Aktuelle Frage: "{query}"

Schreibe einen kompakten Absatz (max. 4 Sätze) der NUR die für diese Frage
relevanten Informationen enthält. Keine Aufzählung, natürlicher Fließtext.
Wenn nichts relevant ist: "Keine relevanten Brain-Einträge."
"""

_KOMPRIMIER_CACHE_TTL_S = 300   # 5 Minuten Cache
_KOMPRIMIER_MIN_ENTRIES = 3     # Nur komprimieren wenn genug Entries da


class SemanticMemory:
    def __init__(
        self,
        brain_manager: "BrainManager",
        brain_index: "BrainIndex",
        hybrid: "HybridSearch | None" = None,
        ollama: "OllamaClient | None" = None,
        nano_modell: str = "qwen2.5:3b",
    ) -> None:
        self._manager    = brain_manager
        self._index      = brain_index
        self._hybrid     = hybrid
        self._ollama     = ollama
        self._nano       = nano_modell
        self._cache: dict[str, tuple[str, float]] = {}   # hash → (text, ts)
        self._letzte_entry_ids: list[str] = []            # v3.3: für Feedback-Loop

    def set_hybrid(self, hybrid: "HybridSearch") -> None:
        self._hybrid = hybrid

    def set_ollama(self, ollama: "OllamaClient", nano_modell: str = "qwen2.5:3b") -> None:
        """Setzt OllamaClient für LLM-Komprimierung (nach Startup aufrufbar)."""
        self._ollama  = ollama
        self._nano    = nano_modell

    async def suche(
        self,
        q_vec: list[float],
        query_text: str = "",
        max_ergebnisse: int = 5,
        min_score: float = 0.0,
        min_vertrauen: float = 0.5,
    ) -> list["BrainEntry"]:
        """Gibt relevante Brain-Entries zurück.

        Nutzt HybridSearch wenn verfügbar, sonst Fallback auf reinen Vector-Search.
        Erhöht abruf_count für jeden Treffer (fire-and-forget).

        Args:
            q_vec:          Embedding-Vektor der Anfrage.
            query_text:     Raw-Text der Anfrage (für BM25-Keyword-Matching).
            max_ergebnisse: Maximale Anzahl zurückgegebener Entries.
            min_score:      Minimaler Cosinus-Score (nur für Fallback-Modus).
            min_vertrauen:  Entries unter diesem Wert werden gefiltert.
        """
        if not q_vec and not query_text:
            return []

        ergebnis: list["BrainEntry"] = []

        if self._hybrid is not None and (q_vec or query_text):
            # ── Hybrid-Modus (Predator) ───────────────────────────────
            alle = await self._manager.alle()
            treffer = self._hybrid.suche(
                q_vec       = q_vec,
                query_text  = query_text,
                alle_entries = alle,
                max_ergebnisse = max_ergebnisse,
                min_vertrauen  = min_vertrauen,
            )
            for entry_id, _ in treffer:
                entry = await self._manager.get(entry_id)
                if entry:
                    ergebnis.append(entry)
                    # Abruf registrieren (fire-and-forget — kein await-Fehler wenn ausbleibt)
                    try:
                        await self._manager.abruf_incrementieren(entry_id)
                    except Exception:
                        pass
        else:
            # ── Fallback: reiner Vector-Search (originale Logik) ─────
            if not q_vec:
                return []
            treffer_vec = self._index.suche(
                q_vec, max_ergebnisse=max_ergebnisse * 2, min_score=min_score
            )
            for entry_id, score in treffer_vec:
                entry = await self._manager.get(entry_id)
                if entry and entry.vertrauen >= min_vertrauen:
                    ergebnis.append(entry)
                    try:
                        await self._manager.abruf_incrementieren(entry_id)
                    except Exception:
                        pass
                if len(ergebnis) >= max_ergebnisse:
                    break

        log.debug(
            "SemanticMemory: %d Treffer (hybrid=%s, query='%s')",
            len(ergebnis), self._hybrid is not None, query_text[:30]
        )
        self._letzte_entry_ids = [e.id for e in ergebnis]
        return ergebnis

    @property
    def letzte_entry_ids(self) -> list[str]:
        """IDs der zuletzt gefundenen Brain-Entries — für Feedback-Loop."""
        return list(self._letzte_entry_ids)

    async def als_fakten_text(
        self,
        q_vec: list[float],
        query_text: str = "",
        max_ergebnisse: int = 5,
    ) -> list[str]:
        """Gibt Brain-Treffer als Liste von Fact-Strings zurück.

        v3.3: Holt 15 Kandidaten intern. Wenn OllamaClient gesetzt ist und
        genug Einträge vorliegen, komprimiert das Nano-LLM die 15 Treffer zu
        einem prägnanten Absatz (gecached 5 Min). Sonst: Top-5 wie vorher.
        """
        # Immer 15 Kandidaten holen für bessere Abdeckung
        kandidaten_n = max(15, max_ergebnisse)
        entries = await self.suche(
            q_vec          = q_vec,
            query_text     = query_text,
            max_ergebnisse = kandidaten_n,
            min_vertrauen  = 0.5,
        )

        if not entries:
            return []

        # Ohne LLM oder zu wenig Einträge: Top-N direkt zurückgeben
        if not self._ollama or len(entries) < _KOMPRIMIER_MIN_ENTRIES:
            return [e.inhalt for e in entries[:max_ergebnisse]]

        # LLM-Komprimierung mit Cache
        fakten_text = "\n".join(f"- {e.inhalt}" for e in entries)
        cache_key   = hashlib.md5(
            f"{query_text}|{fakten_text}".encode()
        ).hexdigest()

        # Cache-Treffer?
        if cache_key in self._cache:
            cached_text, cached_ts = self._cache[cache_key]
            if time.monotonic() - cached_ts < _KOMPRIMIER_CACHE_TTL_S:
                log.debug("SemanticMemory: Brain-Komprimierung aus Cache")
                return [cached_text]

        # LLM komprimieren
        try:
            prompt = _KOMPRIMIER_PROMPT.format(
                fakten=fakten_text[:2000],
                query=query_text[:200],
            )
            komprimiert = await self._ollama.chat(
                nachrichten=[{"role": "user", "content": prompt}],
                modell=self._nano,
                optionen={"temperature": 0.1, "num_predict": 300, "think": False},
            )
            komprimiert = komprimiert.strip()
            if komprimiert and "Keine relevanten" not in komprimiert:
                self._cache[cache_key] = (komprimiert, time.monotonic())
                log.debug("SemanticMemory: Brain komprimiert (%d→1 Absatz)", len(entries))
                return [komprimiert]
        except Exception as e:
            log.debug("Brain-Komprimierung fehlgeschlagen: %s — fallback", e)

        # Fallback: rohe Top-N
        return [e.inhalt for e in entries[:max_ergebnisse]]
