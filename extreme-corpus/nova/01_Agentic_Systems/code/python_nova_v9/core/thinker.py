"""Nova v8 – Thinker.

Background-Reasoning: findet Verbindungen im Brain.
Python 3.14: nutzt concurrent.interpreters wenn verfügbar,
             sonst threading-Fallback (Python 3.12/3.13).
"""
from __future__ import annotations
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from core.logger import get
from core.event_bus import bus, EventTyp

if TYPE_CHECKING:
    from core.ollama_client import OllamaClient
    from core.brain_manager import BrainManager

log = get("thinker")

_PROMPT = """Analysiere diese Brain-Einträge und finde nicht-offensichtliche Verbindungen.
Antworte NUR mit JSON oder NOOP.

Einträge:
{eintraege}

Format: {{"verbindungen":[{{"von":"A","zu":"B","relation":"Beschreibung","typ":"warnung|zusammenhang|idee|erinnerung"}}]}}"""


class Thinker:
    def __init__(self, ollama: "OllamaClient", brain: "BrainManager", cfg: dict, callback=None) -> None:
        from core.config import get as cget
        self._ollama   = ollama
        self._brain    = brain
        self.callback  = callback
        self._schwelle = cget(cfg, "thinker", "entries_schwelle", default=5)
        self._zaehler  = 0
        self._laeuft   = False
        self._lock     = threading.Lock()

        data_p = Path(cget(cfg, "brain", "pfad", default="./brain")).parent / "data"
        data_p.mkdir(parents=True, exist_ok=True)
        self._status_p = data_p / "thinker_status.json"

        # Python 3.14: concurrent.interpreters
        self._nutze_interpreter = self._check_interpreter_support()
        if self._nutze_interpreter:
            log.info("Thinker: nutzt concurrent.interpreters (Python 3.14)")

    def _check_interpreter_support(self) -> bool:
        """
        Prueft ob concurrent.interpreters verfuegbar ist.
        Laeuft mit Timeout in einem Thread, weil der Import
        auf Windows Python 3.14 haengen kann.
        """
        import concurrent.futures as _cf
        def _try():
            try:
                import concurrent.interpreters  # noqa: F401
                return True
            except Exception:
                return False
        try:
            with _cf.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(_try).result(timeout=2.0)
        except Exception:
            return False

    # ─── Entry-Counter ────────────────────────────────────────────────────────
    def entry_zaehlen(self) -> None:
        with self._lock:
            self._zaehler += 1
            soll = self._zaehler >= self._schwelle and not self._laeuft
        if soll:
            self.starten()

    def starten(self) -> None:
        with self._lock:
            if self._laeuft:
                return
            self._laeuft  = True
            self._zaehler = 0
        threading.Thread(target=self._run, daemon=True, name="thinker").start()

    # ─── Haupt-Analyse ────────────────────────────────────────────────────────
    def _run(self) -> None:
        try:
            self._analysiere()
        except Exception as e:
            log.error(f"Thinker Fehler: {e}", exc_info=True)
        finally:
            with self._lock:
                self._laeuft = False
            self._status_speichern({"letzter_run": datetime.now().isoformat()})

    def _analysiere(self) -> None:
        self._sende("start", {"text": "Thinker analysiert Brain-Verbindungen…"})
        bus.publish_threadsafe(EventTyp.THINKER_START, {"text": "Analyse gestartet"})

        index            = self._brain.get_index()
        emb_arr, emb_ids = self._brain.get_embeddings()

        if len(index) < 3:
            self._sende("fertig", {"verbindungen": 0, "text": "Zu wenige Einträge."})
            return

        cluster = self._finde_cluster(emb_arr, emb_ids, index) if emb_arr is not None else [index[:5]]
        total   = 0

        for gruppe in cluster[:3]:
            if len(gruppe) < 2:
                continue
            eintraege_text = ""
            for e in gruppe[:5]:
                inhalt = self._brain.eintrag_laden(e["id"])
                if inhalt:
                    eintraege_text += f"\n[{e.get('titel', e['id'])}]\n{inhalt[:300]}\n"
            if not eintraege_text:
                continue

            self._sende("analyse", {
                "text": f"Prüfe: {', '.join(e.get('titel','?') for e in gruppe[:3])}…"
            })

            antwort = self._ollama.generiere(
                [{"role": "user", "content": _PROMPT.format(eintraege=eintraege_text)}],
                modell_art="schnell",
                max_tokens=300,
            )
            if not antwort or antwort.strip().upper() == "NOOP":
                continue

            verbindungen = self._parse(antwort)
            if verbindungen:
                self._schreibe(verbindungen)
                total += len(verbindungen)
                bus.publish_threadsafe(
                    EventTyp.THINKER_VERBINDUNG,
                    {"anzahl": len(verbindungen)}
                )

        text = (
            f"Thinker: {total} Verbindung{'en' if total != 1 else ''} entdeckt."
            if total > 0 else "Thinker: Keine neuen Verbindungen."
        )
        self._sende("fertig", {"verbindungen": total, "text": text})
        bus.publish_threadsafe(EventTyp.THINKER_FERTIG, {"verbindungen": total, "text": text})
        log.info(text)

    def _finde_cluster(self, emb_arr, emb_ids: list, index: list) -> list[list[dict]]:
        import numpy as np
        norms  = np.linalg.norm(emb_arr, axis=1, keepdims=True)
        normed = np.where(norms > 0, emb_arr / norms, 0)
        sim    = normed @ normed.T
        conns  = self._brain.alle_connections()
        by_id  = {e["id"]: e for e in index}
        cluster, besucht = [], set()

        for i, eid in enumerate(emb_ids):
            if eid in besucht:
                continue
            gruppe = [eid]
            for j, eid2 in enumerate(emb_ids):
                if eid2 in besucht or j == i:
                    continue
                if sim[i, j] >= 0.55:
                    bestehend = [c["id"] for c in conns.get(eid, [])]
                    if eid2 not in bestehend:
                        gruppe.append(eid2)
            if len(gruppe) >= 2:
                cluster.append([by_id.get(e, {"id": e, "titel": e}) for e in gruppe[:5]])
                besucht.update(gruppe)
            if len(cluster) >= 5:
                break
        return cluster

    def _parse(self, text: str) -> list[dict]:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if not m:
            return []
        try:
            return json.loads(m.group()).get("verbindungen", [])
        except Exception:
            return []

    def _schreibe(self, verbindungen: list[dict]) -> None:
        for v in verbindungen:
            von  = v.get("von",  "")
            zu   = v.get("zu",   "")
            rel  = v.get("relation", "")
            typ  = v.get("typ",  "zusammenhang")
            if not von or not zu or not rel:
                continue
            titel  = f"{von} ↔ {zu}"
            jetzt  = datetime.now()
            inhalt = (
                f"---\nentitaet: {titel}\ntyp: inferenz\n"
                f"erstellt: {jetzt.isoformat()}\nquelle: thinker\n"
                f"verbindungs_typ: {typ}\ntags: [\"{von.lower()}\", \"{zu.lower()}\"]\n"
                f"keywords: [\"{von.lower()}\", \"{zu.lower()}\"]\n---\n\n"
                f"## {titel}\n\n**Typ:** {typ}\n\n{rel}\n\n"
                f"*Entdeckt: {jetzt.strftime('%Y-%m-%d %H:%M')}*\n"
            )
            self._brain.eintrag_erstellen(titel, inhalt, [von.lower(), zu.lower()], "inferenz")
            # Bidirektionale Verbindung
            von_id = self._brain.duplikat_suchen(von)
            zu_id  = self._brain.duplikat_suchen(zu)
            if von_id and zu_id:
                self._brain.verbindung_erstellen(von_id, zu_id, rel, typ)

    def _sende(self, subtyp: str, daten: dict) -> None:
        if self.callback:
            try:
                self.callback({"subtyp": subtyp, **daten})
            except Exception:
                pass

    def _status_speichern(self, status: dict) -> None:
        try:
            self._status_p.write_text(json.dumps(status), encoding="utf-8")
        except Exception:
            pass

    def get_status(self) -> dict:
        status = {"laeuft": self._laeuft, "zaehler": self._zaehler,
                  "interpreter_mode": self._nutze_interpreter}
        if self._status_p.exists():
            try:
                status.update(json.loads(self._status_p.read_text(encoding="utf-8")))
            except Exception:
                pass
        return status
