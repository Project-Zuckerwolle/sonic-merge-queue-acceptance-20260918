"""Nova Predator v1 — BrainManager.
CRUD für BrainEntry JSON-Dateien. Kein Nova-Import außer logger.
Graph-Struktur: Einträge mit bidirektionalen Links.

Predator-Upgrades:
  - BrainLink: typisierte Verbindungen (verwandt, erweitert, widerspricht, ...)
  - BrainEntry: 4 neue Felder (gate_score, gewicht, abruf_count, veraltet)
  - typed_links: neues Feld neben links (rückwärts-kompatibel)
  - abruf_incrementieren(): wird von SemanticMemory nach Treffer aufgerufen
"""
from __future__ import annotations
import json
import uuid
import dataclasses
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import aiofiles

from core.logger import get

log = get("brain_manager")

BrainTyp = Literal["fakt", "praeferenz", "aufgabe", "idee", "verbindung", "notiz", "durchbruch"]

# Typen für typisierte Verbindungen zwischen Brain-Entries
BrainLinkTyp = Literal["verwandt", "erweitert", "widerspricht", "folgt_aus", "beispiel_fuer", "migriert"]


@dataclass
class BrainLink:
    """Typisierte Verbindung zwischen zwei Brain-Entries.

    Wird in typed_links gespeichert — separates Feld von links (str-Liste).
    Rückwärts-kompatibel: bestehende links-Einträge bleiben unverändert.
    """
    ziel_id: str
    typ: BrainLinkTyp = "verwandt"
    grund: str = ""
    staerke: float = 0.8    # 0.0–1.0

    def zu_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def von_dict(cls, d: dict) -> "BrainLink":
        erlaubte = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in erlaubte})


@dataclass
class BrainEntry:
    id: str
    typ: BrainTyp
    inhalt: str
    quelle: str                        # 'chat', 'thinker', 'manuell', 'apex'
    tags: list[str] = field(default_factory=list)
    vertrauen: float = 0.8             # 0.0–1.0
    vektor: list[float] | None = None  # Embedding-Vektor
    links: list[str] = field(default_factory=list)  # IDs verlinkter Entries (legacy, bleibt)
    erstellt: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    zuletzt_bestaetigt: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    score: float = 0.0                 # Ideen-Score (0.0–1.0)

    # ── Predator-Felder (alle mit Default → rückwärts-kompatibel) ────
    gate_score: float = 0.0            # BrainGate-Bewertung (0–10); 0 = alt/unbewertet
    gewicht: str = "vollstaendig"      # "vollstaendig" | "notiz"
    abruf_count: int = 0               # wie oft per SemanticSearch abgerufen
    veraltet: bool = False             # True wenn durch TemporalUpdate invalidiert
    typed_links: list = field(default_factory=list)  # list[BrainLink] — typisierte Verbindungen

    # ── v3.1-Felder ──────────────────────────────────────────────────
    conversation_id: str | None = None  # Gespräch in dem dieser Fakt entstand (None = vor v3.1)

    def zu_dict(self) -> dict:
        d = asdict(self)
        # typed_links korrekt serialisieren (asdict behandelt dataclasses rekursiv)
        return d

    @classmethod
    def von_dict(cls, d: dict) -> "BrainEntry":
        erlaubte = {f.name for f in dataclasses.fields(cls)}
        gefiltert = {k: v for k, v in d.items() if k in erlaubte}
        # typed_links: list[dict] → list[BrainLink] deserialisieren
        if "typed_links" in gefiltert and gefiltert["typed_links"]:
            raw = gefiltert["typed_links"]
            gefiltert["typed_links"] = [
                BrainLink.von_dict(item) if isinstance(item, dict) else item
                for item in raw
            ]
        return cls(**gefiltert)


class BrainManager:
    def __init__(self, brain_pfad: str | Path = "brain") -> None:
        self._pfad = Path(brain_pfad)
        self._pfad.mkdir(parents=True, exist_ok=True)
        self._index_datei = self._pfad / "_index.json"
        self._cache: dict[str, BrainEntry] = {}
        self._geladen = False

    def _entry_pfad(self, entry_id: str) -> Path:
        return self._pfad / f"{entry_id}.json"

    async def laden(self) -> None:
        """Lädt alle Entries in den Cache."""
        self._cache.clear()
        for datei in self._pfad.glob("*.json"):
            if datei.name.startswith("_"):
                continue
            try:
                async with aiofiles.open(datei, encoding="utf-8") as f:
                    daten = json.loads(await f.read())
                entry = BrainEntry.von_dict(daten)
                self._cache[entry.id] = entry
            except Exception as e:
                log.warning("Brain Entry %s konnte nicht geladen werden: %s", datei.name, e)
        self._geladen = True
        log.info("Brain geladen: %d Einträge", len(self._cache))

    async def add(self, entry: BrainEntry) -> BrainEntry:
        """Fügt neuen Entry hinzu oder aktualisiert bestehenden."""
        if not entry.id:
            entry.id = str(uuid.uuid4())
        self._cache[entry.id] = entry
        await self._speichern(entry)
        log.debug("Brain Entry gespeichert: %s (%s)", entry.id[:8], entry.typ)
        return entry

    async def get(self, entry_id: str) -> BrainEntry | None:
        return self._cache.get(entry_id)

    async def alle(self) -> list[BrainEntry]:
        return list(self._cache.values())

    async def suche_text(self, query: str, max_ergebnisse: int = 10) -> list[BrainEntry]:
        """Einfache Textsuche im Inhalt."""
        q = query.lower()
        treffer = [e for e in self._cache.values() if q in e.inhalt.lower()]
        return treffer[:max_ergebnisse]

    async def seit(self, zeitstempel: str) -> list[BrainEntry]:
        """Gibt alle Entries zurück die nach zeitstempel erstellt wurden."""
        return [e for e in self._cache.values() if e.erstellt > zeitstempel]

    async def loeschen(self, entry_id: str) -> bool:
        if entry_id not in self._cache:
            return False
        # Aus verlinkten Entries (links + typed_links) entfernen
        entry = self._cache[entry_id]
        for link_id in entry.links:
            if link_id in self._cache:
                self._cache[link_id].links = [l for l in self._cache[link_id].links if l != entry_id]
                # typed_links ebenfalls bereinigen
                self._cache[link_id].typed_links = [
                    tl for tl in self._cache[link_id].typed_links
                    if (tl.ziel_id if isinstance(tl, BrainLink) else tl.get("ziel_id")) != entry_id
                ]
                await self._speichern(self._cache[link_id])
        del self._cache[entry_id]
        datei = self._entry_pfad(entry_id)
        if datei.exists():
            datei.unlink()
        return True

    async def add_link(self, von_id: str, zu_id: str) -> bool:
        """Erstellt bidirektionalen Link zwischen zwei Entries (legacy links-Liste)."""
        if von_id not in self._cache or zu_id not in self._cache:
            return False
        if zu_id not in self._cache[von_id].links:
            self._cache[von_id].links.append(zu_id)
            await self._speichern(self._cache[von_id])
        if von_id not in self._cache[zu_id].links:
            self._cache[zu_id].links.append(von_id)
            await self._speichern(self._cache[zu_id])
        return True

    async def add_typed_link(
        self,
        von_id: str,
        zu_id: str,
        typ: BrainLinkTyp = "verwandt",
        grund: str = "",
        staerke: float = 0.8,
    ) -> bool:
        """Erstellt typisierten bidirektionalen Link (typed_links).
        Schreibt auch in legacy links-Liste für Rückwärtskompatibilität.
        """
        if von_id not in self._cache or zu_id not in self._cache:
            return False

        def _hat_typed_link(entry: BrainEntry, ziel_id: str) -> bool:
            return any(
                (tl.ziel_id if isinstance(tl, BrainLink) else tl.get("ziel_id")) == ziel_id
                for tl in entry.typed_links
            )

        if not _hat_typed_link(self._cache[von_id], zu_id):
            self._cache[von_id].typed_links.append(
                BrainLink(ziel_id=zu_id, typ=typ, grund=grund, staerke=staerke)
            )
            await self._speichern(self._cache[von_id])

        # Bidirektional: Gegenrichtung mit neutralem Typ
        if not _hat_typed_link(self._cache[zu_id], von_id):
            self._cache[zu_id].typed_links.append(
                BrainLink(ziel_id=von_id, typ=typ, grund=grund, staerke=staerke)
            )
            await self._speichern(self._cache[zu_id])

        # Legacy-Links synchron halten
        await self.add_link(von_id, zu_id)
        return True

    async def abruf_incrementieren(self, entry_id: str) -> None:
        """Erhöht abruf_count wenn Entry per SemanticSearch gefunden wurde.
        Wird von SemanticMemory nach jedem Treffer aufgerufen.
        Kein Fehler wenn Entry nicht gefunden (fire-and-forget).
        """
        entry = self._cache.get(entry_id)
        if entry is None:
            return
        entry.abruf_count += 1
        # Vertrauen leicht erhöhen — häufig abgerufene Entries sind relevant
        entry.vertrauen = min(1.0, entry.vertrauen + 0.02)
        await self._speichern(entry)

    async def vertrauen_anpassen(self, entry_id: str, delta: float) -> bool:
        if entry_id not in self._cache:
            return False
        entry = self._cache[entry_id]
        entry.vertrauen = max(0.0, min(1.0, entry.vertrauen + delta))
        entry.zuletzt_bestaetigt = datetime.now(timezone.utc).isoformat()
        await self._speichern(entry)
        return True

    def nach_typ(self, typ: BrainTyp) -> list[BrainEntry]:
        return [e for e in self._cache.values() if e.typ == typ]

    def stats(self) -> dict:
        alle = list(self._cache.values())
        aktiv = [e for e in alle if not e.veraltet]
        return {
            "gesamt":       len(alle),
            "aktiv":        len(aktiv),
            "veraltet":     len(alle) - len(aktiv),
            "typen": {
                t: sum(1 for e in aktiv if e.typ == t)
                for t in ("fakt", "praeferenz", "aufgabe", "idee", "verbindung", "notiz", "durchbruch")
            },
            "avg_gate_score": round(
                sum(e.gate_score for e in aktiv if e.gate_score > 0) /
                max(sum(1 for e in aktiv if e.gate_score > 0), 1), 2
            ),
            "mit_typed_links": sum(1 for e in aktiv if e.typed_links),
        }

    async def _speichern(self, entry: BrainEntry) -> None:
        datei = self._entry_pfad(entry.id)
        async with aiofiles.open(datei, "w", encoding="utf-8") as f:
            await f.write(json.dumps(entry.zu_dict(), ensure_ascii=False, indent=2))
