"""Nova v8 – Episodic Memory.

Episodisches Gedächtnis: Sessions, persistente Embeddings auf Disk.
FIX aus v7: Embeddings überleben Neustart (memory_embeddings.npy).
Semantische + chronologische Retrieval.
"""
from __future__ import annotations
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from core.logger import get
from core.config import get as cget

if TYPE_CHECKING:
    from core.ollama_client import OllamaClient

log = get("episodic")


class EpisodicMemory:
    def __init__(self, cfg: dict, ollama: "OllamaClient") -> None:
        self._ollama    = ollama
        self._sess_p    = Path(cget(cfg, "memory", "session_pfad",  default="./memory/sessions"))
        self._archiv_p  = Path(cget(cfg, "memory", "archiv_pfad",   default="./memory/archives"))
        self._max_ctx   = cget(cfg, "memory", "max_kontext_nachrichten", default=8)
        self._dauer_h   = cget(cfg, "memory", "session_dauer_stunden",   default=24)
        self._emb_datei = Path(cget(cfg, "memory", "embed_datei",    default="./data/memory_embeddings.npy"))
        self._emb_ids_d = Path(cget(cfg, "memory", "embed_ids_datei",default="./data/memory_embed_ids.json"))

        self._sess_p.mkdir(parents=True, exist_ok=True)
        self._archiv_p.mkdir(parents=True, exist_ok=True)
        self._emb_datei.parent.mkdir(parents=True, exist_ok=True)

        self._eintraege: list[dict] = []
        self._emb_arr: np.ndarray | None = None
        self._emb_ids: list[int] = []

        self._lade_session()
        self._lade_embeddings()

    # ─── Session laden/speichern ──────────────────────────────────────────────
    def _lade_session(self) -> None:
        p = self._sess_p / "current.json"
        if not p.exists():
            self._eintraege = []
            return
        try:
            daten = json.loads(p.read_text(encoding="utf-8"))
            # Auto-Archivierung bei Ablauf
            erstellt = datetime.fromisoformat(daten.get("erstellt", datetime.now().isoformat()))
            if datetime.now() - erstellt > timedelta(hours=self._dauer_h):
                log.info("Session abgelaufen – archiviere")
                self._archiviere(daten)
                self._eintraege = []
            else:
                self._eintraege = daten.get("eintraege", [])
                log.info(f"Session geladen: {len(self._eintraege)} Einträge")
        except Exception as e:
            log.error(f"Session-Ladefehler: {e}")
            self._eintraege = []

    def _speichere_session(self) -> None:
        p = self._sess_p / "current.json"
        try:
            # Erstellt-Zeitstempel aus erstem Eintrag oder jetzt
            erstellt = self._eintraege[0]["zeit"] if self._eintraege else datetime.now().isoformat()
            daten = {"erstellt": erstellt, "eintraege": self._eintraege}
            p.write_text(json.dumps(daten, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            log.error(f"Session-Speicherfehler: {e}")

    def _archiviere(self, daten: dict) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = self._archiv_p / f"session_{ts}.json"
        try:
            p.write_text(json.dumps(daten, ensure_ascii=False, indent=2), encoding="utf-8")
            log.info(f"Session archiviert: {p.name}")
        except Exception as e:
            log.error(f"Archivierungsfehler: {e}")

    # ─── Embeddings laden/speichern (persistiert auf Disk) ────────────────────
    def _lade_embeddings(self) -> None:
        """Lädt Embedding-Matrix von Disk (v8-Feature: überlebt Neustart)."""
        try:
            if self._emb_datei.exists() and self._emb_ids_d.exists():
                arr = np.load(str(self._emb_datei))
                ids = json.loads(self._emb_ids_d.read_text(encoding="utf-8"))
                if arr.shape[0] == len(ids):
                    self._emb_arr = arr
                    self._emb_ids = ids
                    log.info(f"Memory-Embeddings geladen: {len(ids)} Vektoren")
                else:
                    log.warning("Memory-Embeddings inkonsistent – neu aufbauen")
                    self._emb_arr = None
                    self._emb_ids = []
        except Exception as e:
            log.warning(f"Memory-Embeddings Ladefehler: {e} – nur chronologisch")

    def _speichere_embeddings(self) -> None:
        """Persistiert Embedding-Matrix auf Disk."""
        if self._emb_arr is None or len(self._emb_ids) == 0:
            return
        try:
            np.save(str(self._emb_datei), self._emb_arr)
            self._emb_ids_d.write_text(json.dumps(self._emb_ids), encoding="utf-8")
        except Exception as e:
            log.error(f"Memory-Embedding-Speicherfehler: {e}")

    # ─── Eintrag hinzufügen ───────────────────────────────────────────────────
    def hinzufuegen(self, rolle: str, inhalt: str) -> None:
        """Fügt Eintrag hinzu und berechnet Embedding."""
        idx = len(self._eintraege)
        eintrag = {
            "idx": idx,
            "rolle": rolle,
            "inhalt": inhalt,
            "zeit": datetime.now().isoformat(),
        }
        self._eintraege.append(eintrag)
        self._speichere_session()

        # Embedding berechnen + persistieren
        vec = self._ollama.embed(inhalt)
        if vec is not None:
            neuer_arr = np.array([vec], dtype=np.float32)
            if self._emb_arr is None:
                self._emb_arr = neuer_arr
            else:
                self._emb_arr = np.vstack([self._emb_arr, neuer_arr])
            self._emb_ids.append(idx)
            self._speichere_embeddings()

    # ─── Retrieval ────────────────────────────────────────────────────────────
    def baue_kontext(self, q_vec: list[float] | None, max_n: int | None = None) -> list[dict]:
        """Gibt relevante Session-Einträge zurück."""
        n = max_n or self._max_ctx
        if not self._eintraege:
            return []
        if q_vec is not None and self._emb_arr is not None and len(self._emb_ids) > 0:
            return self._semantisch(q_vec, n)
        return self._chronologisch(n)

    def _semantisch(self, q_vec: list[float], n: int) -> list[dict]:
        qv = np.array(q_vec, dtype=np.float32)
        qn = np.linalg.norm(qv)
        if qn == 0:
            return self._chronologisch(n)
        qv = qv / qn
        norms = np.linalg.norm(self._emb_arr, axis=1, keepdims=True)
        normed = np.where(norms > 0, self._emb_arr / norms, 0)
        sims = normed @ qv
        top_idx = np.argsort(sims)[::-1][:n]
        ergebnis = []
        for i in top_idx:
            if i < len(self._emb_ids):
                eidx = self._emb_ids[i]
                if eidx < len(self._eintraege):
                    e = dict(self._eintraege[eidx])
                    e["aehnlichkeit"] = round(float(sims[i]), 3)
                    ergebnis.append(e)
        return sorted(ergebnis, key=lambda x: x.get("idx", 0))

    def _chronologisch(self, n: int) -> list[dict]:
        return list(self._eintraege[-n:])

    # ─── Komprimierung / Archivierung ─────────────────────────────────────────
    def komprimieren(self) -> dict:
        """Archiviert aktuelle Session, startet neue."""
        daten = {"erstellt": (self._eintraege[0]["zeit"] if self._eintraege else datetime.now().isoformat()),
                 "eintraege": self._eintraege}
        self._archiviere(daten)
        self._eintraege = []
        self._emb_arr = None
        self._emb_ids = []
        # Embedding-Dateien löschen
        for p in [self._emb_datei, self._emb_ids_d]:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        self._speichere_session()
        log.info("Session komprimiert und neu gestartet")
        return {"archiviert": len(daten["eintraege"])}

    def archiv_liste(self) -> list[str]:
        return sorted([p.name for p in self._archiv_p.glob("session_*.json")], reverse=True)

    # ─── Status ───────────────────────────────────────────────────────────────
    def status(self) -> dict:
        return {
            "eintraege": len(self._eintraege),
            "mit_embedding": len(self._emb_ids),
            "archiviert": len(self.archiv_liste()),
        }
