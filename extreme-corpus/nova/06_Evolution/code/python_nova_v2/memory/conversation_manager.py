"""Nova Predator v3.1 — ConversationManager.

Zentrale Verwaltung aller Gespräche (Conversations).
Ersetzt den alten einfachen `session_*.json` Archiv-Mechanismus durch ein
strukturiertes System mit Gespräch-IDs, Titeln und Metadaten.

Jedes Gespräch wird in `memory/conversations/conv_YYYYMMDD_HHMMSS.json`
gespeichert. Das Format ist:

    {
      "id": "conv_20260422_141530",
      "erstellt": "2026-04-22T14:15:30Z",
      "aktualisiert": "2026-04-22T15:42:11Z",
      "aktiv": true,              // aktuell geöffnet?
      "titel": "Hammerfall Planung",
      "turns": [
        {"rolle": "user", "inhalt": "...", "zeitstempel": "..."},
        {"rolle": "assistant", "inhalt": "...", "zeitstempel": "..."}
      ],
      "session_facts": [
        {"text": "...", "typ": "fakt", "konfidenz": 0.9, "zeitstempel": "..."}
      ]
    }

Invarianten:
  - Es gibt immer maximal 1 aktives Gespräch
  - Gespräche werden NICHT automatisch gelöscht — nur explizit via loesche()
  - Turn-Format kompatibel mit EpisodicMemory.ChatTurn
  - Legacy session_*.json Dateien werden einmalig migriert
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles

from core.logger import get

log = get("conversation_manager")

_CONV_PFAD_DEFAULT = "memory/conversations"
_LEGACY_SESSION_PFAD = "memory/sessions"


# ── Datenstrukturen ──────────────────────────────────────────────────────────

@dataclass
class ConversationTurn:
    """Ein einzelner Turn (User- oder Assistant-Nachricht)."""
    rolle: str  # "user" | "assistant"
    inhalt: str
    zeitstempel: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def zu_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def von_dict(cls, d: dict) -> "ConversationTurn":
        return cls(
            rolle=d.get("rolle", "user"),
            inhalt=d.get("inhalt", ""),
            zeitstempel=d.get("zeitstempel", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class Conversation:
    """Ein vollständiges Gespräch mit Metadaten, Turns und Session-Facts."""
    id: str
    erstellt: str
    aktualisiert: str
    aktiv: bool = False
    titel: str = "Neues Gespräch"
    turns: list[ConversationTurn] = field(default_factory=list)
    session_facts: list[dict] = field(default_factory=list)

    @property
    def turn_anzahl(self) -> int:
        return len(self.turns)

    @property
    def preview(self) -> str:
        """Kurze Vorschau aus dem ersten User-Turn."""
        for t in self.turns:
            if t.rolle == "user":
                return t.inhalt[:100]
        return "(leer)"

    def zu_dict(self) -> dict:
        return {
            "id": self.id,
            "erstellt": self.erstellt,
            "aktualisiert": self.aktualisiert,
            "aktiv": self.aktiv,
            "titel": self.titel,
            "turns": [t.zu_dict() for t in self.turns],
            "session_facts": self.session_facts,
        }

    @classmethod
    def von_dict(cls, d: dict) -> "Conversation":
        return cls(
            id=d["id"],
            erstellt=d.get("erstellt", datetime.now(timezone.utc).isoformat()),
            aktualisiert=d.get("aktualisiert", d.get("erstellt", "")),
            aktiv=bool(d.get("aktiv", False)),
            titel=d.get("titel", "Gespräch"),
            turns=[ConversationTurn.von_dict(t) for t in d.get("turns", [])],
            session_facts=d.get("session_facts", []),
        )


# ── ConversationManager ──────────────────────────────────────────────────────

class ConversationManager:
    """Verwaltet alle Gespräche im Dateisystem.

    In-Memory-Cache für Performance, Disk-Backup für Persistenz.
    Thread-safe für Einzel-Event-Loop (keine Locks nötig).
    """

    def __init__(self, pfad: str | Path = _CONV_PFAD_DEFAULT) -> None:
        self._pfad = Path(pfad)
        self._cache: dict[str, Conversation] = {}
        self._geladen = False
        log.debug("ConversationManager erstellt: %s", self._pfad)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def laden(self) -> int:
        """Lädt alle conv_*.json Dateien in den Cache.
        Migriert auch alte session_*.json beim ersten Start.

        Returns:
            Anzahl geladener Gespräche.
        """
        self._pfad.mkdir(parents=True, exist_ok=True)

        # Migration: alte session_*.json → conv_*.json (einmalig)
        migriert = await self._migriere_legacy()
        if migriert > 0:
            log.info("Migration: %d alte Sessions konvertiert", migriert)

        # Cache befüllen
        self._cache.clear()
        for datei in self._pfad.glob("conv_*.json"):
            try:
                async with aiofiles.open(datei, encoding="utf-8") as f:
                    daten = json.loads(await f.read())
                conv = Conversation.von_dict(daten)
                self._cache[conv.id] = conv
            except Exception as e:
                log.warning("Gespräch %s konnte nicht geladen werden: %s",
                            datei.name, e)

        self._geladen = True
        log.info("Gespräche geladen: %d", len(self._cache))
        return len(self._cache)

    async def _migriere_legacy(self) -> int:
        """Konvertiert alte session_*.json zu conv_*.json Format.
        Läuft nur wenn conv-Verzeichnis leer UND sessions/ existiert.
        """
        legacy_dir = Path(_LEGACY_SESSION_PFAD)
        if not legacy_dir.exists():
            return 0

        # Nur migrieren wenn noch keine conv-Dateien existieren
        bestehende = list(self._pfad.glob("conv_*.json"))
        if bestehende:
            return 0

        migriert = 0
        for legacy in sorted(legacy_dir.glob("session_*.json")):
            try:
                async with aiofiles.open(legacy, encoding="utf-8") as f:
                    alte_turns = json.loads(await f.read())

                if not isinstance(alte_turns, list):
                    continue

                # Zeitstempel aus Dateiname extrahieren
                ts_match = re.match(r"session_(\d{8}_\d{6})", legacy.stem)
                ts_str = ts_match.group(1) if ts_match else "unknown"

                # Konvertieren in neues Format
                conv_id = f"conv_{ts_str}"
                conv = Conversation(
                    id=conv_id,
                    erstellt=self._parse_ts_aus_string(ts_str),
                    aktualisiert=self._parse_ts_aus_string(ts_str),
                    aktiv=False,
                    titel=self._generiere_titel_aus_turns(alte_turns),
                    turns=[ConversationTurn.von_dict(t) for t in alte_turns
                           if isinstance(t, dict)],
                    session_facts=[],
                )

                await self._speichern(conv)
                migriert += 1
            except Exception as e:
                log.warning("Legacy-Migration fehlgeschlagen für %s: %s",
                            legacy.name, e)

        return migriert

    @staticmethod
    def _parse_ts_aus_string(ts: str) -> str:
        """Konvertiert '20260422_141530' zu ISO-String."""
        try:
            dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _generiere_titel_aus_turns(turns: list) -> str:
        """Nimmt ersten User-Turn als Titel-Basis."""
        for t in turns:
            if isinstance(t, dict) and t.get("rolle") == "user":
                inhalt = t.get("inhalt", "")
                # Erste 50 Zeichen, ohne Zeilenumbrüche
                titel = inhalt[:50].replace("\n", " ").strip()
                return titel if titel else "Altes Gespräch"
        return "Altes Gespräch"

    # ── CRUD ─────────────────────────────────────────────────────────────────

    async def neue_conversation(self, titel: str = "") -> Conversation:
        """Erstellt ein neues Gespräch und markiert es als aktiv.
        Deaktiviert automatisch das vorher aktive Gespräch.
        """
        # Altes aktives deaktivieren
        await self._deaktiviere_alle()

        ts = datetime.now(timezone.utc)
        ts_str = ts.strftime("%Y%m%d_%H%M%S")
        conv_id = f"conv_{ts_str}"

        # Bei Kollision (sehr selten): UUID-Suffix
        if conv_id in self._cache:
            conv_id = f"conv_{ts_str}_{uuid.uuid4().hex[:6]}"

        conv = Conversation(
            id=conv_id,
            erstellt=ts.isoformat(),
            aktualisiert=ts.isoformat(),
            aktiv=True,
            titel=titel or "Neues Gespräch",
            turns=[],
            session_facts=[],
        )

        self._cache[conv_id] = conv
        await self._speichern(conv)
        log.info("Neues Gespräch: %s", conv_id)
        return conv

    async def wechsle_zu(self, conv_id: str) -> Conversation | None:
        """Wechselt zu einem bestehenden Gespräch.
        Deaktiviert das aktuell aktive.

        Returns:
            Die geladene Conversation oder None wenn unbekannt.
        """
        if conv_id not in self._cache:
            log.warning("wechsle_zu: Unbekannte ID %s", conv_id)
            return None

        await self._deaktiviere_alle()

        conv = self._cache[conv_id]
        conv.aktiv = True
        conv.aktualisiert = datetime.now(timezone.utc).isoformat()
        await self._speichern(conv)
        log.info("Gewechselt zu Gespräch: %s", conv_id)
        return conv

    async def _deaktiviere_alle(self) -> None:
        """Markiert alle aktiven Gespräche als inaktiv."""
        for conv in self._cache.values():
            if conv.aktiv:
                conv.aktiv = False
                await self._speichern(conv)

    def aktive_conversation(self) -> Conversation | None:
        """Gibt das aktuell aktive Gespräch zurück (oder None)."""
        for conv in self._cache.values():
            if conv.aktiv:
                return conv
        return None

    def letzte_conversation(self) -> Conversation | None:
        """Gibt das zuletzt aktualisierte Gespräch zurück (aktiv oder nicht).
        Wird beim Start genutzt wenn kein Gespräch als 'aktiv' markiert ist.
        """
        if not self._cache:
            return None
        return max(self._cache.values(), key=lambda c: c.aktualisiert)

    def get(self, conv_id: str) -> Conversation | None:
        return self._cache.get(conv_id)

    def alle(self) -> list[Conversation]:
        """Alle Gespräche, sortiert nach aktualisiert (neueste zuerst)."""
        return sorted(
            self._cache.values(),
            key=lambda c: c.aktualisiert,
            reverse=True,
        )

    async def speichere_turn(
        self,
        conv_id: str,
        rolle: str,
        inhalt: str,
    ) -> bool:
        """Hängt einen Turn an ein Gespräch und persistiert."""
        conv = self._cache.get(conv_id)
        if not conv:
            log.warning("speichere_turn: Unbekannte ID %s", conv_id)
            return False

        turn = ConversationTurn(rolle=rolle, inhalt=inhalt)
        conv.turns.append(turn)
        conv.aktualisiert = turn.zeitstempel

        # Automatischer Titel nach erstem User-Turn
        if conv.titel in ("Neues Gespräch", "", "Gespräch") and rolle == "user":
            conv.titel = inhalt[:50].replace("\n", " ").strip() or "Neues Gespräch"

        await self._speichern(conv)
        return True

    async def speichere_session_facts(
        self,
        conv_id: str,
        facts: list[dict],
    ) -> bool:
        """Speichert die aktuellen Session-Facts im Gespräch."""
        conv = self._cache.get(conv_id)
        if not conv:
            return False
        conv.session_facts = facts
        conv.aktualisiert = datetime.now(timezone.utc).isoformat()
        await self._speichern(conv)
        return True

    async def loesche(self, conv_id: str) -> bool:
        """Löscht ein Gespräch vollständig."""
        if conv_id not in self._cache:
            return False
        conv = self._cache[conv_id]
        war_aktiv = conv.aktiv
        datei = self._pfad / f"{conv_id}.json"
        try:
            if datei.exists():
                datei.unlink()
        except Exception as e:
            log.warning("Konnte %s nicht löschen: %s", datei, e)
            return False
        del self._cache[conv_id]
        log.info("Gespräch gelöscht: %s", conv_id)

        # Wenn aktives gelöscht wurde: letzte verbleibende als aktiv markieren
        if war_aktiv:
            letzte = self.letzte_conversation()
            if letzte:
                letzte.aktiv = True
                await self._speichern(letzte)
        return True

    # ── Intern ───────────────────────────────────────────────────────────────

    async def _speichern(self, conv: Conversation) -> None:
        """Schreibt eine Conversation auf Disk."""
        datei = self._pfad / f"{conv.id}.json"
        try:
            async with aiofiles.open(datei, "w", encoding="utf-8") as f:
                await f.write(json.dumps(
                    conv.zu_dict(), ensure_ascii=False, indent=2,
                ))
        except Exception as e:
            log.error("Konnte %s nicht speichern: %s", datei, e)

    def status(self) -> dict[str, Any]:
        aktiv = self.aktive_conversation()
        return {
            "anzahl": len(self._cache),
            "aktiv_id": aktiv.id if aktiv else None,
            "aktiv_titel": aktiv.titel if aktiv else None,
            "aktiv_turns": aktiv.turn_anzahl if aktiv else 0,
            "pfad": str(self._pfad),
        }
