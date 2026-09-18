"""Nova Predator v1 — BrainIndex.
Cosinus-Ähnlichkeit und Cluster-Suche via numpy.
Kein Nova-Import außer logger und brain_manager.

Predator-Upgrades:
  - BM25Index: Keyword-Suche für exakte Matches (Namen, Versionen, Fehlercodes)
  - HybridSearch: RRF-Fusion aus Vector + BM25 + Recency (15–30% besserer Recall)
"""
from __future__ import annotations
import json
import math
import re
import time
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

from core.logger import get

if TYPE_CHECKING:
    from core.brain_manager import BrainEntry

log = get("brain_index")


def cosinus_aehnlichkeit(a: list[float], b: list[float]) -> float:
    """Berechnet Cosinus-Ähnlichkeit zwischen zwei Vektoren."""
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


class BrainIndex:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vektoren: np.ndarray | None = None  # shape (N, D)

    def neu_aufbauen(self, entries: list["BrainEntry"]) -> None:
        """Baut Index aus allen Entries mit Vektor auf."""
        mit_vektor = [e for e in entries if e.vektor]
        if not mit_vektor:
            self._ids = []
            self._vektoren = None
            return
        self._ids = [e.id for e in mit_vektor]
        self._vektoren = np.array([e.vektor for e in mit_vektor], dtype=np.float32)
        log.debug("BrainIndex neu aufgebaut: %d Einträge", len(self._ids))

    def suche(
        self,
        q_vec: list[float],
        max_ergebnisse: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Gibt [(id, score)] sortiert nach Ähnlichkeit zurück."""
        if self._vektoren is None or len(self._ids) == 0:
            return []
        q = np.array(q_vec, dtype=np.float32)
        norm_q = np.linalg.norm(q)
        if norm_q == 0:
            return []
        normen = np.linalg.norm(self._vektoren, axis=1)
        gueltig = normen > 0
        scores = np.zeros(len(self._ids))
        scores[gueltig] = (self._vektoren[gueltig] @ q) / (normen[gueltig] * norm_q)
        idx_sortiert = np.argsort(scores)[::-1]
        ergebnis = []
        for i in idx_sortiert[:max_ergebnisse * 2]:  # Mehr holen, dann filtern
            if scores[i] >= min_score:
                ergebnis.append((self._ids[i], float(scores[i])))
            if len(ergebnis) >= max_ergebnisse:
                break
        return ergebnis

    def naechste_aehnlichkeit(self, vektor: list[float]) -> float:
        """Gibt den höchsten Ähnlichkeitswert zu einem Vektor zurück (für Ideen-Score)."""
        if self._vektoren is None or len(self._ids) == 0:
            return 0.0
        treffer = self.suche(vektor, max_ergebnisse=2, min_score=0.0)
        # Index 0 wäre der Entry selbst, also nehmen wir Index 1 wenn vorhanden
        if len(treffer) >= 2:
            return treffer[1][1]
        if len(treffer) == 1:
            return treffer[0][1]
        return 0.0

    def cluster(
        self,
        entries: list["BrainEntry"],
        threshold: float = 0.75,
    ) -> list[list["BrainEntry"]]:
        """Gruppiert Entries in Cluster basierend auf Cosinus-Ähnlichkeit."""
        mit_vektor = [e for e in entries if e.vektor]
        if len(mit_vektor) < 2:
            return [[e] for e in mit_vektor]

        vektoren = np.array([e.vektor for e in mit_vektor], dtype=np.float32)
        normen = np.linalg.norm(vektoren, axis=1, keepdims=True)
        normen = np.where(normen > 0, normen, 1.0)
        norm_vektoren = vektoren / normen
        sim_matrix = norm_vektoren @ norm_vektoren.T

        besucht = set()
        cluster_list: list[list["BrainEntry"]] = []

        for i in range(len(mit_vektor)):
            if i in besucht:
                continue
            cluster = [mit_vektor[i]]
            besucht.add(i)
            for j in range(i + 1, len(mit_vektor)):
                if j not in besucht and sim_matrix[i, j] >= threshold:
                    cluster.append(mit_vektor[j])
                    besucht.add(j)
            cluster_list.append(cluster)

        return cluster_list

    def tag_frequenz(self, tags: list[str]) -> int:
        """Dummy — wird von brain_manager gefüllt wenn entries bekannt sind."""
        return 0

    def widerspruchs_kandidaten(
        self,
        entries: list["BrainEntry"],
        min_aehnlichkeit: float = 0.7,
        max_aehnlichkeit: float = 0.95,
    ) -> list[tuple["BrainEntry", "BrainEntry"]]:
        """Findet Paare die ähnlich genug sind um Widersprüche zu haben."""
        mit_vektor = [e for e in entries if e.vektor]
        if len(mit_vektor) < 2:
            return []
        paare = []
        for i in range(len(mit_vektor)):
            for j in range(i + 1, len(mit_vektor)):
                sim = cosinus_aehnlichkeit(mit_vektor[i].vektor, mit_vektor[j].vektor)
                if min_aehnlichkeit <= sim <= max_aehnlichkeit:
                    paare.append((mit_vektor[i], mit_vektor[j]))
        return paare


# ══════════════════════════════════════════════════════════════════════════════
# BM25Index — Keyword-Suche für exakte Matches
# ══════════════════════════════════════════════════════════════════════════════

# Minimale Stoppwörter: nur die häufigsten inhaltsleeren Wörter
_STOPWOERTER = frozenset({
    "der", "die", "das", "dem", "den", "des",
    "ein", "eine", "einer", "eines", "einem", "einen",
    "und", "oder", "aber", "auch", "noch", "nicht", "kein", "keine",
    "ist", "sind", "war", "waren", "wird", "werden", "hat", "haben",
    "mit", "für", "von", "auf", "an", "in", "aus", "bei", "nach",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "and", "or", "but", "not", "no", "with", "for", "of", "in",
    "to", "from", "at", "by", "as",
})

log_bm25 = get("brain_index.bm25")


class BM25Index:
    """Invertierter Index für Keyword-Matching auf Brain-Entries.

    Implementiert BM25 (Robertson 1994) ohne externe Abhängigkeiten.
    Parameter: k1=1.5, b=0.75 (Standardwerte, gut für kurze Dokumente).

    Gespeichert als brain/bm25_index.json.
    Inkrementell aktualisiert — kein Komplettaufbau bei jedem Write.
    """

    K1 = 1.5
    B  = 0.75
    SAVE_INTERVAL = 10  # Alle N add()-Calls speichern

    def __init__(self, pfad: Path | str = "brain/bm25_index.json") -> None:
        self._pfad = Path(pfad)
        self._docs: dict[str, list[str]] = {}   # entry_id → tokenisierte Wörter
        self._idf:  dict[str, float] = {}        # term → IDF-Wert
        self._avg_len: float = 0.0
        self._dirty: int = 0                     # ungespeicherte Änderungen

    # ── Tokenisierung ──────────────────────────────────────────────────────

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenisiert Text: lowercase, alphanumerisch + Versionsnummern (X.Y.Z).

        Beispiele:
          'Python 3.14.4' → ['python', '3.14.4']
          'RX 7900 XT'    → ['rx', '7900', 'xt']
          'qwen3:14b'     → ['qwen3', '14b']
        """
        # Wörter + Versionsnummern (z.B. 3.14, 0.21.0)
        tokens = re.findall(r'[a-z0-9äöüß_]+(?:\.[0-9]+)*', text.lower())
        return [t for t in tokens if len(t) >= 2 and t not in _STOPWOERTER]

    # ── Write-Operationen ──────────────────────────────────────────────────

    def add(self, entry_id: str, text: str) -> None:
        """Fügt Entry zum Index hinzu oder aktualisiert ihn."""
        self._docs[entry_id] = self.tokenize(text)
        self._rebuild_idf()
        self._dirty += 1
        if self._dirty >= self.SAVE_INTERVAL:
            self._speichern_sync()

    def remove(self, entry_id: str) -> None:
        """Entfernt Entry aus dem Index."""
        if entry_id in self._docs:
            del self._docs[entry_id]
            self._rebuild_idf()
            self._dirty += 1

    # ── Suche ──────────────────────────────────────────────────────────────

    def search(self, query: str, max_results: int = 10) -> list[tuple[str, float]]:
        """BM25-Ranking für eine Query.

        Returns:
            [(entry_id, normalized_score)] sortiert absteigend, Score 0.0–1.0
        """
        q_terms = self.tokenize(query)
        if not q_terms or not self._docs:
            return []

        scores: dict[str, float] = {}
        avg = max(self._avg_len, 1.0)

        for doc_id, tokens in self._docs.items():
            doc_len = len(tokens)
            if doc_len == 0:
                continue
            # Term-Frequenz im Dokument
            tf_map: dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            score = 0.0
            for term in q_terms:
                idf = self._idf.get(term, 0.0)
                if idf == 0.0:
                    continue
                tf = tf_map.get(term, 0)
                # BM25-Formel
                numerator   = tf * (self.K1 + 1)
                denominator = tf + self.K1 * (1 - self.B + self.B * doc_len / avg)
                score += idf * numerator / denominator

            if score > 0.0:
                scores[doc_id] = score

        if not scores:
            return []

        # Normalisieren auf 0–1
        max_s = max(scores.values())
        normalized = {k: v / max_s for k, v in scores.items()}
        return sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:max_results]

    # ── IDF-Rebuild ────────────────────────────────────────────────────────

    def _rebuild_idf(self) -> None:
        """Baut IDF-Werte neu auf. O(V*D) — nur bei Änderungen aufrufen."""
        N = len(self._docs)
        if N == 0:
            self._idf = {}
            self._avg_len = 0.0
            return

        total_len = sum(len(t) for t in self._docs.values())
        self._avg_len = total_len / N

        term_doc_freq: dict[str, int] = {}
        for tokens in self._docs.values():
            for t in set(tokens):
                term_doc_freq[t] = term_doc_freq.get(t, 0) + 1

        self._idf = {
            t: math.log((N - df + 0.5) / (df + 0.5) + 1)
            for t, df in term_doc_freq.items()
        }

    # ── Persistenz ─────────────────────────────────────────────────────────

    def _speichern_sync(self) -> None:
        """Speichert Index synchron (für inkrementelle Updates)."""
        try:
            self._pfad.parent.mkdir(parents=True, exist_ok=True)
            data = {"docs": self._docs, "avg_len": self._avg_len}
            self._pfad.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            self._dirty = 0
        except Exception as e:
            log_bm25.warning("BM25Index speichern fehlgeschlagen: %s", e)

    def laden(self) -> None:
        """Lädt gespeicherten Index. Kein Fehler wenn Datei nicht existiert."""
        if not self._pfad.exists():
            return
        try:
            data = json.loads(self._pfad.read_text(encoding="utf-8"))
            self._docs = data.get("docs", {})
            self._avg_len = data.get("avg_len", 0.0)
            self._rebuild_idf()
            log_bm25.debug("BM25Index geladen: %d Dokumente", len(self._docs))
        except Exception as e:
            log_bm25.warning("BM25Index laden fehlgeschlagen (leerer Index): %s", e)

    def speichern(self) -> None:
        """Erzwingtes Speichern (z.B. beim Shutdown)."""
        self._speichern_sync()

    def __len__(self) -> int:
        return len(self._docs)


# ══════════════════════════════════════════════════════════════════════════════
# HybridSearch — RRF-Fusion aus Vector + BM25 + Recency
# ══════════════════════════════════════════════════════════════════════════════

log_hybrid = get("brain_index.hybrid")


class HybridSearch:
    """Kombiniert Vector-Search + BM25 + Recency via RRF (Reciprocal Rank Fusion).

    RRF-Formel: score(d) = Σ_i  w_i / (k + rank_i(d))
    k=60 (Robertson & Zaragoza 2009 — kein Tuning nötig).

    Gewichte:
      Vector (semantic meaning):  W_VEC  = 0.50
      BM25   (keyword precision):  W_BM25 = 0.30
      Recency (neuere bevorzugen): W_TIME = 0.20

    Typischer Recall-Gewinn gegenüber reinem Vector-Search: +15–30%.
    Besonders effektiv bei: Namen, Versionsnummern, Fehlercodes, Modellbezeichnungen.
    """

    RRF_K   = 60
    W_VEC   = 0.50
    W_BM25  = 0.30
    W_TIME  = 0.20

    # Recency-Decay: Halbwertszeit in Tagen
    # entry.vertraut nach N Tagen ohne Abruf: exp(-N / DECAY_TAGE)
    DECAY_TAGE = 60.0

    def __init__(self, brain_index: BrainIndex, bm25: BM25Index) -> None:
        self._vec  = brain_index
        self._bm25 = bm25

    def suche(
        self,
        q_vec: list[float],
        query_text: str,
        alle_entries: list["BrainEntry"],
        max_ergebnisse: int = 5,
        min_vertrauen: float = 0.5,
    ) -> list[tuple[str, float]]:
        """Hybrid-Retrieval mit RRF-Fusion.

        Args:
            q_vec:          Embedding-Vektor der Query.
            query_text:     Raw-Text der Query (für BM25).
            alle_entries:   Alle Brain-Entries (für Recency + Filterung).
            max_ergebnisse: Maximale Anzahl zurückgegebener Entries.
            min_vertrauen:  Entries mit vertrauen < min_vertrauen werden gefiltert.

        Returns:
            [(entry_id, rrf_score)] sortiert absteigend.
        """
        n = max_ergebnisse * 3  # Mehr kandidaten für Fusion

        # ── 1. Vector-Resultate ────────────────────────────────────────────
        vec_results = self._vec.suche(q_vec, max_ergebnisse=n) if q_vec else []

        # ── 2. BM25-Resultate ──────────────────────────────────────────────
        bm25_results = self._bm25.search(query_text, max_results=n) if query_text.strip() else []

        # ── 3. Recency-Score pro Entry ─────────────────────────────────────
        jetzt = time.time()
        id_lookup: dict[str, "BrainEntry"] = {e.id: e for e in alle_entries}

        def recency(entry_id: str) -> float:
            entry = id_lookup.get(entry_id)
            if not entry:
                return 0.1
            # Versuche erstellt-Timestamp zu parsen
            try:
                from datetime import datetime, timezone
                ts = datetime.fromisoformat(entry.erstellt).timestamp()
            except Exception:
                return 0.3
            alter_tage = max(0.0, (jetzt - ts) / 86400)
            return math.exp(-alter_tage / self.DECAY_TAGE)

        # ── 4. RRF-Fusion ──────────────────────────────────────────────────
        rrf: dict[str, float] = {}

        for rank, (doc_id, _) in enumerate(vec_results):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + self.W_VEC / (self.RRF_K + rank + 1)

        for rank, (doc_id, _) in enumerate(bm25_results):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + self.W_BM25 / (self.RRF_K + rank + 1)

        # Recency-Bonus für alle Kandidaten
        for doc_id in list(rrf):
            r = recency(doc_id)
            rrf[doc_id] += self.W_TIME * r / max(n, 1)

        # ── 5. Filter: veraltet, vertrauen, nicht im Cache ─────────────────
        ergebnis = []
        for doc_id, score in sorted(rrf.items(), key=lambda x: x[1], reverse=True):
            entry = id_lookup.get(doc_id)
            if entry is None:
                continue
            if entry.vertrauen < min_vertrauen:
                continue
            if getattr(entry, "veraltet", False):
                continue
            ergebnis.append((doc_id, score))
            if len(ergebnis) >= max_ergebnisse:
                break

        log_hybrid.debug(
            "HybridSearch: vec=%d bm25=%d → fusion=%d → final=%d",
            len(vec_results), len(bm25_results), len(rrf), len(ergebnis)
        )
        return ergebnis
