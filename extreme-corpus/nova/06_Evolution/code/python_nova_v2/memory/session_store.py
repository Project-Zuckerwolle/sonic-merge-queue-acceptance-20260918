"""Nova Predator v2 — SessionStore.
TTL-basierter Cache für Session-Facts.
Ermöglicht Facts-Restore nach Browser-Reload ohne Brain-Commit.

Design:
  - Session-ID wird beim ersten WS-Connect generiert
  - Facts werden alle 60s automatisch gespeichert (Scheduler-Job)
  - Bei Reconnect mit gültiger Session-ID: vollständige Restore
  - TTL: 8 Stunden (konfigurierbar)
  - Maximale Sessions gleichzeitig: 10 (älteste werden verdrängt)

Kein LLM, kein Ollama — reiner In-Memory-Cache.
"""
from __future__ import annotations
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from core.logger import get

log = get("session_store")

_DEFAULT_TTL_S  = 8 * 3600   # 8 Stunden
_MAX_SESSIONS   = 10


@dataclass
class _SessionEntry:
    session_id: str
    facts: list[dict]          # serialisierte SessionFact-Objekte
    erstellt: float            # time.monotonic()
    aktualisiert: float


class SessionStore:
    """In-Memory TTL-Cache für Session-Facts.

    Verwendung:
        store = SessionStore()
        sid = store.neue_session()
        store.speichere(sid, session.session_facts.alle())
        facts = store.lade(sid)   # None wenn abgelaufen oder unbekannt
    """

    def __init__(self, ttl_s: int = _DEFAULT_TTL_S) -> None:
        self._ttl_s = ttl_s
        self._sessions: dict[str, _SessionEntry] = {}

    # ── Public API ────────────────────────────────────────────────────

    def neue_session(self) -> str:
        """Erzeugt neue Session-ID und registriert sie."""
        self._aufraumen()
        sid = str(uuid.uuid4())
        jetzt = time.monotonic()
        self._sessions[sid] = _SessionEntry(
            session_id=sid,
            facts=[],
            erstellt=jetzt,
            aktualisiert=jetzt,
        )
        log.debug("Neue Session: %s", sid[:8])
        return sid

    def speichere(self, session_id: str, facts: list[Any]) -> bool:
        """Speichert SessionFacts für eine Session.

        Args:
            facts: Liste von SessionFact-Objekten (mit .text, .typ, .konfidenz)
        Returns:
            True wenn erfolgreich, False wenn Session unbekannt/abgelaufen.
        """
        self._aufraumen()
        entry = self._sessions.get(session_id)
        if not entry:
            log.debug("speichere: Session %s unbekannt", session_id[:8])
            return False

        # Serialisiere Facts als Dicts (für JSON-Kompatibilität)
        serialisiert = []
        for f in facts:
            if hasattr(f, "text"):
                serialisiert.append({
                    "text": f.text,
                    "typ": getattr(f, "typ", "fakt"),
                    "konfidenz": getattr(f, "konfidenz", 0.9),
                })
            elif isinstance(f, dict):
                serialisiert.append(f)

        entry.facts = serialisiert
        entry.aktualisiert = time.monotonic()
        log.debug("Session %s: %d Facts gespeichert", session_id[:8], len(serialisiert))
        return True

    def lade(self, session_id: str) -> list[dict] | None:
        """Lädt Facts für eine Session.

        Returns:
            Liste von Fact-Dicts oder None wenn nicht gefunden/abgelaufen.
        """
        self._aufraumen()
        entry = self._sessions.get(session_id)
        if not entry:
            return None
        log.debug("Session %s restored: %d Facts", session_id[:8], len(entry.facts))
        return list(entry.facts)

    def ist_gueltig(self, session_id: str) -> bool:
        """Prüft ob eine Session-ID gültig und nicht abgelaufen ist."""
        self._aufraumen()
        return session_id in self._sessions

    def loeschen(self, session_id: str) -> None:
        """Entfernt eine Session explizit."""
        self._sessions.pop(session_id, None)

    def aktive_sessions(self) -> int:
        """Gibt Anzahl aktiver (nicht-abgelaufener) Sessions zurück."""
        self._aufraumen()
        return len(self._sessions)

    # ── Internes ──────────────────────────────────────────────────────

    def _aufraumen(self) -> None:
        """Entfernt abgelaufene Sessions. Verdrängt älteste wenn > MAX."""
        jetzt = time.monotonic()
        abgelaufen = [
            sid for sid, entry in self._sessions.items()
            if (jetzt - entry.aktualisiert) > self._ttl_s
        ]
        for sid in abgelaufen:
            del self._sessions[sid]
            log.debug("Session %s abgelaufen und entfernt", sid[:8])

        # Wenn immer noch zu viele: älteste entfernen
        if len(self._sessions) > _MAX_SESSIONS:
            nach_alter = sorted(
                self._sessions.items(),
                key=lambda x: x[1].aktualisiert,
            )
            zu_entfernen = len(self._sessions) - _MAX_SESSIONS
            for sid, _ in nach_alter[:zu_entfernen]:
                del self._sessions[sid]
                log.debug("Session %s verdrängt (MAX erreicht)", sid[:8])
