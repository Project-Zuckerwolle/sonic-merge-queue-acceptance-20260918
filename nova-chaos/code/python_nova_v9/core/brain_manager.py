"""Nova v8 – BrainManager v2.

Persistenter Wissensspeicher. Upgrades gegenüber v7:
- Atomare Entries (1 Eintrag = 1 Konzept)
- Bidirektionale Links (A→B immer auch B→A)
- Tag-Index für schnelle Filterung
- Python 3.14 kompatibel
"""
from __future__ import annotations
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from core.logger import get
from core.config import get as cget

log = get("brain")

TYPEN = frozenset({
    "fahrzeug", "person", "projekt", "praeferenz", "technik",
    "termin", "ort", "finanzen", "tier", "gesundheit",
    "inferenz", "sonstiges", "notiz", "wissen", "aufgabe",
})


def _sicherer_name(titel: str) -> str:
    name = re.sub(r'[^\w\s-]', '', titel.lower())
    name = re.sub(r'[\s]+', '_', name.strip())
    return name[:60] or "eintrag"


class BrainManager:
    def __init__(self, cfg: dict) -> None:
        self._pfad        = Path(cget(cfg, "brain", "pfad",              default="./brain"))
        self._conn_datei  = Path(cget(cfg, "brain", "connections_datei", default="./data/_connections.json"))
        self._tags_datei  = Path(cget(cfg, "brain", "tags_datei",        default="./data/_tags.json"))
        self._fuzzy_schw  = cget(cfg, "brain", "fuzzy_duplikat_schwelle",default=85)
        self._emb_datei   = Path("./data/brain_embeddings.npy")
        self._emb_ids_d   = Path("./data/brain_embed_ids.json")
        self._index_datei = self._pfad / "_index.json"

        self._pfad.mkdir(parents=True, exist_ok=True)
        self._pfad.parent.joinpath("data").mkdir(parents=True, exist_ok=True)

        self._index:       list[dict] = []
        self._connections: dict[str, list[dict]] = {}
        self._tags:        dict[str, list[str]]  = {}  # tag → [entry_ids]
        self._emb_arr:     np.ndarray | None = None
        self._emb_ids:     list[str] = []

        self._lade_index()
        self._lade_connections()
        self._lade_tags()
        self._lade_embeddings()

    # ─── Index ────────────────────────────────────────────────────────────────
    def _lade_index(self) -> None:
        if self._index_datei.exists():
            try:
                self._index = json.loads(self._index_datei.read_text(encoding="utf-8"))
            except Exception as e:
                log.error(f"Index-Ladefehler: {e}")
                self._index = []
        else:
            self._index = []

    def _speichere_index(self) -> None:
        try:
            self._index_datei.write_text(
                json.dumps(self._index, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            log.error(f"Index-Speicherfehler: {e}")

    # ─── Connections (jetzt bidirektional) ────────────────────────────────────
    def _lade_connections(self) -> None:
        if self._conn_datei.exists():
            try:
                self._connections = json.loads(self._conn_datei.read_text(encoding="utf-8"))
            except Exception:
                self._connections = {}

    def _speichere_connections(self) -> None:
        self._conn_datei.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn_datei.write_text(
                json.dumps(self._connections, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            log.error(f"Connections-Speicherfehler: {e}")

    def _verbindung_hinzu(self, von_id: str, zu_id: str, relation: str, typ: str) -> None:
        """Fügt bidirektionale Verbindung hinzu."""
        def _add(quell_id, ziel_id, rel):
            beste = self._connections.setdefault(quell_id, [])
            if not any(c["id"] == ziel_id for c in beste):
                beste.append({"id": ziel_id, "relation": rel, "typ": typ})

        _add(von_id, zu_id, relation)
        _add(zu_id, von_id, f"↔ {relation}")  # Rückrichtung
        self._speichere_connections()

    # ─── Tags ─────────────────────────────────────────────────────────────────
    def _lade_tags(self) -> None:
        if self._tags_datei.exists():
            try:
                self._tags = json.loads(self._tags_datei.read_text(encoding="utf-8"))
            except Exception:
                self._tags = {}

    def _speichere_tags(self) -> None:
        self._tags_datei.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._tags_datei.write_text(
                json.dumps(self._tags, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            log.error(f"Tags-Speicherfehler: {e}")

    def _tags_indexieren(self, entry_id: str, tags: list[str]) -> None:
        for tag in tags:
            self._tags.setdefault(tag.lower(), [])
            if entry_id not in self._tags[tag.lower()]:
                self._tags[tag.lower()].append(entry_id)
        self._speichere_tags()

    # ─── Embeddings ───────────────────────────────────────────────────────────
    def _lade_embeddings(self) -> None:
        try:
            if self._emb_datei.exists() and self._emb_ids_d.exists():
                arr = np.load(str(self._emb_datei))
                ids = json.loads(self._emb_ids_d.read_text(encoding="utf-8"))
                if arr.shape[0] == len(ids):
                    self._emb_arr = arr
                    self._emb_ids = ids
                    log.info(f"Brain-Embeddings geladen: {len(ids)}")
        except Exception as e:
            log.warning(f"Embedding-Ladefehler: {e}")

    def _speichere_embeddings(self) -> None:
        if self._emb_arr is None:
            return
        try:
            np.save(str(self._emb_datei), self._emb_arr)
            self._emb_ids_d.write_text(json.dumps(self._emb_ids), encoding="utf-8")
        except Exception as e:
            log.error(f"Embedding-Speicherfehler: {e}")

    # ─── CRUD ─────────────────────────────────────────────────────────────────
    def eintrag_erstellen(
        self, titel: str, inhalt: str,
        keywords: list[str] | None = None,
        kategorie: str = "sonstiges",
        tags: list[str] | None = None,
    ) -> str:
        kat = kategorie if kategorie in TYPEN else "sonstiges"
        entry_id = _sicherer_name(titel)
        # Duplikat-Check
        if self.duplikat_suchen(titel):
            existing = self.duplikat_suchen(titel)
            log.debug(f"Duplikat gefunden für '{titel}' → {existing}")
            return existing

        datei = self._pfad / f"{entry_id}.txt"
        # Unique ID wenn Datei schon existiert
        if datei.exists():
            ts = datetime.now().strftime("%H%M%S")
            entry_id = f"{entry_id}_{ts}"
            datei = self._pfad / f"{entry_id}.txt"

        jetzt = datetime.now().isoformat()
        kws = keywords or []
        tgs = tags or []
        frontmatter = (
            f"---\n"
            f"entitaet: {titel}\n"
            f"typ: {kat}\n"
            f"erstellt: {jetzt}\n"
            f"quelle: conversation\n"
            f"tags: {json.dumps(tgs, ensure_ascii=False)}\n"
            f"keywords: {json.dumps(kws, ensure_ascii=False)}\n"
            f"---\n\n"
        )
        datei.write_text(frontmatter + inhalt, encoding="utf-8")

        meta: dict[str, Any] = {
            "id": entry_id, "titel": titel, "typ": kat,
            "erstellt": jetzt, "tags": tgs, "keywords": kws,
        }
        self._index.append(meta)
        self._speichere_index()
        self._tags_indexieren(entry_id, tgs + kws)
        log.info(f"Brain-Entry erstellt: {entry_id}")
        return entry_id

    def eintrag_laden(self, entry_id: str) -> str | None:
        datei = self._pfad / f"{entry_id}.txt"
        if datei.exists():
            try:
                return datei.read_text(encoding="utf-8")
            except Exception:
                return None
        return None

    def eintrag_updaten(self, entry_id: str, anhang: str, neue_keywords: list[str] | None = None) -> str:
        datei = self._pfad / f"{entry_id}.txt"
        if not datei.exists():
            return entry_id
        jetzt = datetime.now().strftime("%Y-%m-%d %H:%M")
        log_zeile = f"\n### Log\n[{jetzt}] {anhang}\n"
        try:
            inhalt = datei.read_text(encoding="utf-8")
            datei.write_text(inhalt + log_zeile, encoding="utf-8")
        except Exception as e:
            log.error(f"Update-Fehler {entry_id}: {e}")
        if neue_keywords:
            for meta in self._index:
                if meta["id"] == entry_id:
                    meta["keywords"] = list(set(meta.get("keywords", []) + neue_keywords))
                    break
            self._speichere_index()
            self._tags_indexieren(entry_id, neue_keywords)
        return entry_id

    def eintrag_loeschen(self, entry_id: str) -> bool:
        datei = self._pfad / f"{entry_id}.txt"
        if not datei.exists():
            return False
        try:
            datei.unlink()
            self._index = [e for e in self._index if e["id"] != entry_id]
            self._speichere_index()
            self._connections.pop(entry_id, None)
            self._speichere_connections()
            # Aus Embeddings entfernen
            if entry_id in self._emb_ids:
                idx = self._emb_ids.index(entry_id)
                self._emb_ids.pop(idx)
                self._emb_arr = np.delete(self._emb_arr, idx, axis=0) if self._emb_arr is not None else None
                self._speichere_embeddings()
            return True
        except Exception as e:
            log.error(f"Lösch-Fehler {entry_id}: {e}")
            return False

    def duplikat_suchen(self, titel: str) -> str | None:
        titel_lower = titel.lower()
        # Exakter Match zuerst
        for meta in self._index:
            if meta.get("titel", "").lower() == titel_lower:
                return meta["id"]
        # Fuzzy (rapidfuzz falls verfügbar)
        try:
            from rapidfuzz import fuzz
            for meta in self._index:
                if fuzz.ratio(titel_lower, meta.get("titel", "").lower()) >= self._fuzzy_schw:
                    return meta["id"]
        except ImportError:
            pass
        return None

    # ─── Suche ────────────────────────────────────────────────────────────────
    def suche_semantisch(self, q_vec: list[float], top_n: int = 5) -> list[dict]:
        if self._emb_arr is None or len(self._emb_ids) == 0:
            return []
        qv = np.array(q_vec, dtype=np.float32)
        qn = np.linalg.norm(qv)
        if qn == 0:
            return []
        qv = qv / qn
        norms = np.linalg.norm(self._emb_arr, axis=1, keepdims=True)
        normed = np.where(norms > 0, self._emb_arr / norms, 0)
        sims = normed @ qv
        top_idx = np.argsort(sims)[::-1][:top_n]
        ergebnis = []
        by_id = {m["id"]: m for m in self._index}
        for i in top_idx:
            if i < len(self._emb_ids):
                eid = self._emb_ids[i]
                meta = by_id.get(eid, {"id": eid})
                ergebnis.append({**meta, "aehnlichkeit": round(float(sims[i]), 3)})
        return ergebnis

    def suche_fuzzy(self, keywords: list[str], top_n: int = 5) -> list[dict]:
        if not keywords:
            return []
        ergebnis = []
        for meta in self._index:
            meta_kws = " ".join(meta.get("keywords", []) + [meta.get("titel", "")]).lower()
            score = sum(1 for kw in keywords if kw.lower() in meta_kws)
            if score > 0:
                ergebnis.append({**meta, "fuzzy_score": score})
        return sorted(ergebnis, key=lambda x: x["fuzzy_score"], reverse=True)[:top_n]

    def suche_nach_tags(self, tags: list[str]) -> list[dict]:
        ids: set[str] = set()
        for tag in tags:
            ids.update(self._tags.get(tag.lower(), []))
        by_id = {m["id"]: m for m in self._index}
        return [by_id[i] for i in ids if i in by_id]

    # ─── Embeddings speichern ─────────────────────────────────────────────────
    def embedding_speichern(self, entry_id: str, vec: list[float]) -> None:
        v = np.array([vec], dtype=np.float32)
        if entry_id in self._emb_ids:
            idx = self._emb_ids.index(entry_id)
            self._emb_arr[idx] = v[0]
        else:
            self._emb_ids.append(entry_id)
            self._emb_arr = v if self._emb_arr is None else np.vstack([self._emb_arr, v])
        self._speichere_embeddings()

    # ─── Public API ───────────────────────────────────────────────────────────
    def get_embeddings(self) -> tuple[np.ndarray | None, list[str]]:
        return self._emb_arr, self._emb_ids

    def get_index(self) -> list[dict]:
        return list(self._index)

    def get_stats(self) -> dict:
        kategorien: dict[str, int] = {}
        for m in self._index:
            t = m.get("typ", "sonstiges")
            kategorien[t] = kategorien.get(t, 0) + 1
        return {
            "gesamt": len(self._index),
            "mit_embedding": len(self._emb_ids),
            "kategorien": kategorien,
            "tags": len(self._tags),
        }

    def alle_connections(self) -> dict:
        return dict(self._connections)

    def connections_laden(self, entry_id: str) -> list[dict]:
        return list(self._connections.get(entry_id, []))

    def verbindung_erstellen(self, von_id: str, zu_id: str, relation: str, typ: str = "zusammenhang") -> None:
        self._verbindung_hinzu(von_id, zu_id, relation, typ)

    def index_rebuild(self) -> None:
        """Rebuildet Index aus Dateien auf Disk."""
        self._index = []
        for datei in self._pfad.glob("*.txt"):
            if datei.name.startswith("_"):
                continue
            try:
                inhalt = datei.read_text(encoding="utf-8")
                entry_id = datei.stem
                titel = entry_id
                typ = "sonstiges"
                tags: list[str] = []
                kws: list[str] = []
                erstellt = ""
                if inhalt.startswith("---"):
                    fm_end = inhalt.find("---", 3)
                    if fm_end > 0:
                        fm = inhalt[3:fm_end]
                        for line in fm.splitlines():
                            if line.startswith("entitaet:"):
                                titel = line.split(":", 1)[1].strip()
                            elif line.startswith("typ:"):
                                typ = line.split(":", 1)[1].strip()
                            elif line.startswith("erstellt:"):
                                erstellt = line.split(":", 1)[1].strip()
                            elif line.startswith("tags:"):
                                try:
                                    tags = json.loads(line.split(":", 1)[1].strip())
                                except Exception:
                                    pass
                            elif line.startswith("keywords:"):
                                try:
                                    kws = json.loads(line.split(":", 1)[1].strip())
                                except Exception:
                                    pass
                self._index.append({
                    "id": entry_id, "titel": titel, "typ": typ,
                    "erstellt": erstellt, "tags": tags, "keywords": kws,
                })
            except Exception as e:
                log.warning(f"Rebuild-Fehler {datei.name}: {e}")
        self._speichere_index()
        log.info(f"Index rebuild: {len(self._index)} Einträge")
