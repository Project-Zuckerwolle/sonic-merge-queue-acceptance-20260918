"""Nova Predator v2 — OllamaClient.
AsyncClient wrapper für Chat, Embed, Status.
Python 3.14: asyncio.run() / asyncio.get_running_loop() pattern.
Kein Nova-Import außer logger.

v2-Änderungen:
  - VRAM-Semaphore: verhindert gleichzeitiges Laden großer Modelle
  - geladene_modelle(): gibt aktuell im VRAM befindliche Modelle zurück
  - vram_warten(): wartet bis VRAM unter Zielgrenze fällt (für Handoff-Gate)
"""
from __future__ import annotations
import asyncio
import time
from typing import Any, AsyncIterator

from ollama import AsyncClient, ResponseError

from core.logger import get

log = get("ollama_client")

# Semaphore: nur 1 großes Modell darf gleichzeitig laden/entladen
# Verhindert Race Condition zwischen Chat-LLM und Apex/Orchestrator
_VRAM_SEMAPHORE: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Lazy-init des VRAM-Semaphore (braucht laufenden Event-Loop)."""
    global _VRAM_SEMAPHORE
    if _VRAM_SEMAPHORE is None:
        _VRAM_SEMAPHORE = asyncio.Semaphore(1)
    return _VRAM_SEMAPHORE


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 120) -> None:
        self._host = host
        self._timeout = timeout
        # AsyncClient wird lazy erstellt (braucht laufenden Event-Loop)
        self._client: AsyncClient | None = None

    def _get_client(self) -> AsyncClient:
        if self._client is None:
            self._client = AsyncClient(host=self._host, timeout=self._timeout)
        return self._client

    async def ping(self) -> bool:
        """Prüft ob Ollama erreichbar ist."""
        try:
            client = self._get_client()
            await client.list()
            return True
        except Exception as e:
            log.debug("Ollama nicht erreichbar: %s", e)
            return False

    async def modelle(self) -> list[str]:
        """Gibt Liste installierter Modelle zurück."""
        try:
            client = self._get_client()
            result = await client.list()
            return [m.model for m in result.models]
        except Exception as e:
            log.error("Modell-Liste Fehler: %s", e)
            return []

    async def embed(self, text: str, modell: str = "nomic-embed-text") -> list[float]:
        """Erzeugt Embedding-Vektor."""
        try:
            client = self._get_client()
            resp = await client.embed(model=modell, input=text)
            return resp.embeddings[0] if resp.embeddings else []
        except Exception as e:
            log.error("Embed Fehler: %s", e)
            return []

    async def embed_batch(self, texte: list[str], modell: str = "nomic-embed-text") -> list[list[float]]:
        """Erzeugt Embeddings für mehrere Texte."""
        try:
            client = self._get_client()
            resp = await client.embed(model=modell, input=texte)
            return resp.embeddings or []
        except Exception as e:
            log.error("Embed-Batch Fehler: %s", e)
            return []

    async def chat(
        self,
        nachrichten: list[dict[str, str]],
        modell: str = "gemma4:e4b",
        system: str | None = None,
        optionen: dict[str, Any] | None = None,
    ) -> str:
        """Einfacher Chat-Call (kein Streaming). Für Brain-LLM."""
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(nachrichten)
        try:
            client = self._get_client()
            resp = await client.chat(
                model=modell,
                messages=msgs,
                options=optionen or {},
            )
            return resp.message.content or ""
        except ResponseError as e:
            log.error("Chat ResponseError (Modell=%s): %s", modell, e)
            return ""
        except Exception as e:
            log.error("Chat Fehler (Modell=%s): %s", modell, e)
            return ""

    async def stream(
        self,
        nachrichten: list[dict[str, str]],
        modell: str = "gemma4:e4b",
        system: str | None = None,
        optionen: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Streaming Chat — yield token für token."""
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(nachrichten)
        try:
            client = self._get_client()
            async for teil in await client.chat(
                model=modell,
                messages=msgs,
                stream=True,
                options=optionen or {},
            ):
                token = teil.message.content
                if token:
                    yield token
        except ResponseError as e:
            log.error("Stream ResponseError (Modell=%s): %s", modell, e)
        except Exception as e:
            log.error("Stream Fehler (Modell=%s): %s", modell, e)

    async def modell_laden(self, modell: str) -> bool:
        """Lädt ein Modell in Ollama (warm-up). VRAM-Semaphore geschützt."""
        async with _get_semaphore():
            try:
                client = self._get_client()
                await client.chat(
                    model=modell,
                    messages=[{"role": "user", "content": "ping"}],
                    options={"num_predict": 1},
                )
                log.info("Modell '%s' geladen", modell)
                return True
            except Exception as e:
                log.warning("Modell '%s' laden fehlgeschlagen: %s", modell, e)
                return False

    async def modell_entladen(self, modell: str) -> bool:
        """Entlädt Modell aus VRAM (keep_alive=0). VRAM-Semaphore geschützt."""
        async with _get_semaphore():
            try:
                client = self._get_client()
                await client.chat(
                    model=modell,
                    messages=[{"role": "user", "content": ""}],
                    options={"num_predict": 0},
                    keep_alive=0,
                )
                log.info("Modell '%s' entladen", modell)
                return True
            except Exception as e:
                log.warning("Modell '%s' entladen fehlgeschlagen: %s", modell, e)
                return False

    async def geladene_modelle(self) -> list[dict]:
        """Gibt Liste der aktuell im VRAM geladenen Modelle zurück.
        Format: [{"name": str, "vram_gb": float}, ...]
        Nutzt Ollama /api/ps Endpunkt.
        """
        try:
            client = self._get_client()
            result = await client.ps()
            modelle = []
            for m in (result.models or []):
                vram_bytes = getattr(m, "size_vram", 0) or 0
                modelle.append({
                    "name": m.model,
                    "vram_gb": round(vram_bytes / (1024 ** 3), 2),
                })
            return modelle
        except Exception as e:
            log.debug("geladene_modelle Fehler: %s", e)
            return []

    async def vram_warten(
        self,
        ziel_gb: float = 8.0,
        timeout_s: int = 30,
        poll_interval_s: float = 0.5,
    ) -> bool:
        """Wartet bis gesamt-VRAM der geladenen Modelle unter ziel_gb fällt.
        Wird vom OrchestratorGateway nach dem Entladen aufgerufen.
        Gibt True zurück wenn Ziel erreicht, False bei Timeout.
        """
        start = time.monotonic()
        while (time.monotonic() - start) < timeout_s:
            modelle = await self.geladene_modelle()
            gesamt_vram = sum(m["vram_gb"] for m in modelle)
            if gesamt_vram <= ziel_gb:
                log.debug("VRAM unter Ziel (%.1f GB <= %.1f GB)", gesamt_vram, ziel_gb)
                return True
            log.debug("Warte auf VRAM-Freigabe: %.1f GB > %.1f GB", gesamt_vram, ziel_gb)
            await asyncio.sleep(poll_interval_s)
        log.warning("vram_warten Timeout nach %ds (Ziel: %.1f GB)", timeout_s, ziel_gb)
        return False
