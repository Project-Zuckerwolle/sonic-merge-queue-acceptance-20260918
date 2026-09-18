"""Nova v9 – SemanticRouter.

FIX gegenueber v8:
  v8-Problem: Threshold 0.60-0.68 war zu niedrig, alle 10 Skills feuerten
              bei jeder Nachricht (bewiesen durch nova.log).
  v9-Loesung: Top-K Routing mit hartem Mindest-Score.
              - Mindest-Score: 0.72 (config: router.min_score)
              - Maximal 2 Skills gleichzeitig (config: router.max_skills)
              - Kein Skill unter 0.72 = direkte LLM-Antwort ohne Overhead

Routing-Logik:
  1. Alle Skills nach Cosinus-Score sortieren
  2. Nur den/die besten nehmen (max router.max_skills)
  3. Nur wenn Score >= router.min_score
  4. Kein Match = leere Liste = LLM antwortet direkt
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from core.logger import get
from core.config import get as cget

if TYPE_CHECKING:
    from core.ollama_client import OllamaClient

log = get("router")


def _cosinus(a: list[float], b: list[float]) -> float:
    av = np.array(a, dtype=np.float32)
    bv = np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(av), np.linalg.norm(bv)
    return float(np.dot(av, bv) / (na * nb)) if na > 0 and nb > 0 else 0.0


class SemanticRouter:

    def __init__(
        self,
        ollama:      "OllamaClient",
        cfg:         dict,
        cache_pfad:  str = "./data/route_cache.json",
    ) -> None:
        self._ollama    = ollama
        self._min_score = cget(cfg, "router", "min_score",  default=0.72)
        self._max_k     = cget(cfg, "router", "max_skills", default=2)
        self._cache_p   = Path(cache_pfad)
        self._skills:   dict[str, dict] = {}   # name -> {centroid, threshold}
        self._cache     = self._cache_laden()

        log.info(f"Router initialisiert | min_score={self._min_score} | max_skills={self._max_k}")

    # ─── Cache ────────────────────────────────────────────────────────────────

    def _cache_laden(self) -> dict:
        if self._cache_p.exists():
            try:
                return json.loads(self._cache_p.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"skills": {}}

    def _cache_speichern(self) -> None:
        self._cache_p.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._cache_p.write_text(
                json.dumps(self._cache, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            log.error(f"Cache-Speicherfehler: {e}")

    # ─── Initialisierung ──────────────────────────────────────────────────────

    def initialisiere(self, skill_beispiele: dict[str, dict]) -> None:
        """Baut Centroids fuer alle Skills. Nutzt Cache wenn Beispiele unveraendert."""
        if not skill_beispiele:
            log.warning("Router: keine Skill-Beispiele erhalten")
            return

        neu = 0
        for name, info in skill_beispiele.items():
            beispiele = info.get("beispiele", [])
            # Per-Skill threshold aus YAML respektieren,
            # aber auf min_score nach oben deckeln wenn zu niedrig
            yaml_threshold = info.get("threshold", self._min_score)
            threshold = max(yaml_threshold, self._min_score)
            if yaml_threshold < self._min_score:
                log.warning(
                    f"Router: {name} threshold {yaml_threshold} < min_score {self._min_score}"
                    f" - angehoben auf {threshold}"
                )

            if not beispiele:
                log.warning(f"Router: {name} hat keine Beispiele - wird nicht geroutet")
                continue

            # Cache-Check via MD5
            h = hashlib.md5("|".join(sorted(beispiele)).encode()).hexdigest()
            cached = self._cache.get("skills", {}).get(name, {})
            if cached.get("hash") == h and cached.get("centroid"):
                self._skills[name] = {
                    "centroid":  cached["centroid"],
                    "threshold": threshold,
                }
                log.debug(f"Router: {name} aus Cache geladen")
                continue

            # Embeddings berechnen
            log.info(f"Router: Embedde {name} ({len(beispiele)} Beispiele)...")
            vecs = [v for v in [self._ollama.embed(b) for b in beispiele] if v]
            if not vecs:
                log.warning(f"Router: Keine Embeddings fuer {name}")
                continue

            centroid = np.mean(np.array(vecs, dtype=np.float32), axis=0).tolist()
            self._skills[name] = {"centroid": centroid, "threshold": threshold}
            self._cache.setdefault("skills", {})[name] = {
                "centroid":  centroid,
                "threshold": threshold,
                "hash":      h,
            }
            neu += 1

        if neu > 0:
            self._cache_speichern()

        log.info(f"Router bereit: {len(self._skills)} Skills ({neu} neu eingebettet)")

    # ─── Hot-Reload ───────────────────────────────────────────────────────────

    def skill_hinzufuegen(
        self,
        name:      str,
        beispiele: list[str],
        threshold: float | None = None,
    ) -> None:
        """Fuegt einzelnen Skill hinzu (fuer Hot-Reload via inbox/)."""
        vecs = [v for v in [self._ollama.embed(b) for b in beispiele] if v]
        if not vecs:
            log.warning(f"Router: Keine Embeddings fuer neuen Skill {name}")
            return

        thr = max(threshold or self._min_score, self._min_score)
        h   = hashlib.md5("|".join(sorted(beispiele)).encode()).hexdigest()
        centroid = np.mean(np.array(vecs, dtype=np.float32), axis=0).tolist()

        self._skills[name] = {"centroid": centroid, "threshold": thr}
        self._cache.setdefault("skills", {})[name] = {
            "centroid":  centroid,
            "threshold": thr,
            "hash":      h,
        }
        self._cache_speichern()
        log.info(f"Router: '{name}' hinzugefuegt (threshold={thr})")

    def skill_entfernen(self, name: str) -> None:
        self._skills.pop(name, None)
        self._cache.get("skills", {}).pop(name, None)
        self._cache_speichern()
        log.info(f"Router: '{name}' entfernt")

    # ─── Routing ─────────────────────────────────────────────────────────────

    def route(
        self,
        user_input: str,
        q_vec:      list[float] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Top-K Routing: gibt maximal max_skills Skills zurueck,
        nur wenn Score >= threshold des Skills (mind. min_score).

        Gibt leere Liste zurueck wenn kein Skill relevant ist.
        """
        if not self._skills:
            return []

        vec = q_vec or self._ollama.embed(user_input)
        if vec is None:
            log.warning("Router: kein Embedding verfuegbar - kein Skill-Routing")
            return []

        # Alle Scores berechnen
        kandidaten = []
        for name, info in self._skills.items():
            score = _cosinus(vec, info["centroid"])
            threshold = info.get("threshold", self._min_score)
            if score >= threshold:
                kandidaten.append((name, round(score, 3)))

        # Nach Score sortieren, Top-K nehmen
        kandidaten.sort(key=lambda x: x[1], reverse=True)
        treffer = kandidaten[:self._max_k]

        if treffer:
            log.debug(f"Router: {[f'{n}={s}' for n,s in treffer]} (von {len(kandidaten)} Kandidaten)")
        else:
            log.debug(f"Router: kein Skill ueber Threshold (bester: {kandidaten[0] if kandidaten else 'keiner'})")

        return treffer

    # ─── Status ───────────────────────────────────────────────────────────────

    @property
    def ist_bereit(self) -> bool:
        return len(self._skills) > 0

    def status(self) -> dict:
        return {
            "bereit":     self.ist_bereit,
            "skills":     len(self._skills),
            "min_score":  self._min_score,
            "max_skills": self._max_k,
        }
