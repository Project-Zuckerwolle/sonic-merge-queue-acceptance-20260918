"""Nova Predator v1 — EpisodicMemory.
Session-History (Chat-Verlauf) mit Token-Budget und Archivierung.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import aiofiles

from core.logger import get

log = get("episodic_memory")


@dataclass
class ChatTurn:
    rolle: str       # 'user' | 'assistant'
    inhalt: str
    zeitstempel: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def als_llm_nachricht(self) -> dict[str, str]:
        return {"role": self.rolle, "content": self.inhalt}

    def token_schaetzung(self) -> int:
        """Grobe Schätzung: 4 Zeichen ≈ 1 Token."""
        return len(self.inhalt) // 4 + 1


class EpisodicMemory:
    def __init__(
        self,
        max_tokens: int = 8000,
        compress_schwelle: float = 0.85,
        sessions_pfad: str | Path = "memory/sessions",
    ) -> None:
        self._max_tokens = max_tokens
        self._compress_schwelle = compress_schwelle
        self._sessions_pfad = Path(sessions_pfad)
        self._sessions_pfad.mkdir(parents=True, exist_ok=True)
        self._verlauf: list[ChatTurn] = []
        self._token_budget_genutzt = 0

    def add(self, rolle: str, inhalt: str) -> None:
        turn = ChatTurn(rolle=rolle, inhalt=inhalt)
        self._verlauf.append(turn)
        self._token_budget_genutzt += turn.token_schaetzung()

    def als_llm_nachrichten(self, max_tokens: int | None = None) -> list[dict[str, str]]:
        """Gibt die History als LLM-Messages zurück, ggf. gekürzt."""
        limit = max_tokens or self._max_tokens
        nachrichten = []
        tokens = 0
        # Von hinten (neueste zuerst), bis Budget aufgebraucht
        for turn in reversed(self._verlauf):
            t = turn.token_schaetzung()
            if tokens + t > limit:
                break
            nachrichten.insert(0, turn.als_llm_nachricht())
            tokens += t
        return nachrichten

    def token_auslastung(self) -> float:
        """0.0–1.0 wie voll das Token-Budget ist."""
        if self._max_tokens == 0:
            return 0.0
        return min(1.0, self._token_budget_genutzt / self._max_tokens)

    def braucht_kompression(self) -> bool:
        return self.token_auslastung() >= self._compress_schwelle

    def kuerze(self, auf_prozent: float = 0.4) -> int:
        """Entfernt älteste Turns bis Budget auf auf_prozent reduziert.
        Gibt Anzahl entfernter Turns zurück.
        """
        ziel_tokens = int(self._max_tokens * auf_prozent)
        entfernt = 0
        while self._token_budget_genutzt > ziel_tokens and len(self._verlauf) > 2:
            altester = self._verlauf.pop(0)
            self._token_budget_genutzt -= altester.token_schaetzung()
            entfernt += 1
        log.debug("EpisodicMemory gekürzt: %d Turns entfernt", entfernt)
        return entfernt

    def reset(self) -> None:
        self._verlauf.clear()
        self._token_budget_genutzt = 0

    async def archiviere(self) -> Path:
        """Speichert aktuelle Session auf Disk und resettet."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        datei = self._sessions_pfad / f"session_{ts}.json"
        daten = [{"rolle": t.rolle, "inhalt": t.inhalt, "zeitstempel": t.zeitstempel}
                 for t in self._verlauf]
        async with aiofiles.open(datei, "w", encoding="utf-8") as f:
            await f.write(json.dumps(daten, ensure_ascii=False, indent=2))
        log.info("Session archiviert: %s (%d Turns)", datei.name, len(daten))
        self.reset()
        return datei

    def status(self) -> dict:
        return {
            "turns": len(self._verlauf),
            "token_schaetzung": self._token_budget_genutzt,
            "auslastung_prozent": round(self.token_auslastung() * 100, 1),
            "max_tokens": self._max_tokens,
        }
