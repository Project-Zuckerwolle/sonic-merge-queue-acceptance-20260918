"""Nova v8 – Ollama Client.

Multi-Model-Support: verschiedene Modelle für verschiedene Aufgaben.
Chat-Streaming, Embeddings, strukturierte Ausgabe, Retry-Logik.
"""
from __future__ import annotations
import json
from typing import Generator, Any

import requests

from core.logger import get
from core.config import get as cget

log = get("ollama")


class OllamaClient:
    def __init__(self, cfg: dict) -> None:
        self._host    = cget(cfg, "ollama", "host",        default="http://localhost:11434")
        self._temp    = cget(cfg, "ollama", "temperature", default=1.0)
        self._top_p   = cget(cfg, "ollama", "top_p",       default=0.95)
        self._top_k   = cget(cfg, "ollama", "top_k",       default=64)
        self._timeout = 120

        # Modell-Registry aus config
        self._modelle: dict[str, str] = {
            "chat":    cget(cfg, "modelle", "chat",    default=cget(cfg, "ollama", "modell", default="gemma4:e4b")),
            "schnell": cget(cfg, "modelle", "schnell", default="gemma4:e4b"),
            "code":    cget(cfg, "modelle", "code",    default="qwen2.5-coder:7b"),
            "embed":   cget(cfg, "modelle", "embed",   default="nomic-embed-text"),
        }
        log.info(f"Ollama-Client initialisiert: {self._host}")
        log.debug(f"Modelle: {self._modelle}")

    # ─── Modell-Auswahl ───────────────────────────────────────────────────────
    def modell(self, art: str = "chat") -> str:
        """Gibt Modell-Name für die gewünschte Art zurück."""
        return self._modelle.get(art, self._modelle["chat"])

    def modelle_alle(self) -> dict[str, str]:
        return dict(self._modelle)

    # ─── Verfügbarkeit ────────────────────────────────────────────────────────
    def ist_verfuegbar(self) -> bool:
        try:
            r = requests.get(f"{self._host}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def modell_verfuegbar(self, name: str) -> bool:
        try:
            r = requests.get(f"{self._host}/api/tags", timeout=5)
            if r.status_code != 200:
                return False
            modelle = [m["name"] for m in r.json().get("models", [])]
            return any(name in m for m in modelle)
        except Exception:
            return False

    # ─── Embeddings ───────────────────────────────────────────────────────────
    def embed(self, text: str) -> list[float] | None:
        """Embeds Text mit nomic-embed-text. Gibt None bei Fehler."""
        try:
            r = requests.post(
                f"{self._host}/api/embeddings",
                json={"model": self.modell("embed"), "prompt": text},
                timeout=5,  # Schnell fehlschlagen wenn Ollama offline
            )
            if r.status_code == 200:
                return r.json().get("embedding")
            log.debug(f"Embed-Fehler: HTTP {r.status_code}")
            return None
        except Exception as e:
            log.debug(f"Embed-Ausnahme: {e}")
            return None

    # ─── Streaming ────────────────────────────────────────────────────────────
    def stream(
        self,
        messages: list[dict],
        modell_art: str = "chat",
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> Generator[str, None, None]:
        """Streamt Tokens. Yields: einzelne Token-Strings."""
        alle_messages = []
        if system:
            alle_messages.append({"role": "system", "content": system})
        alle_messages.extend(messages)

        payload: dict[str, Any] = {
            "model": self.modell(modell_art),
            "messages": alle_messages,
            "stream": True,
            "options": {
                "temperature": self._temp,
                "top_p": self._top_p,
                "top_k": self._top_k,
            },
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            with requests.post(
                f"{self._host}/api/chat",
                json=payload,
                stream=True,
                timeout=self._timeout,
            ) as resp:
                if resp.status_code != 200:
                    log.error(f"Stream-Fehler: HTTP {resp.status_code}")
                    return
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            log.error(f"Stream-Ausnahme: {e}")

    # ─── Nicht-streaming ──────────────────────────────────────────────────────
    def generiere(
        self,
        messages: list[dict],
        modell_art: str = "chat",
        max_tokens: int = 400,
        system: str | None = None,
    ) -> str | None:
        """Vollständige Antwort (kein Streaming). Gut für strukturierte Outputs."""
        return "".join(self.stream(messages, modell_art, max_tokens, system)) or None

    def generiere_schnell(self, prompt: str, max_tokens: int = 200) -> str | None:
        """Kurzanfrage mit kleinem/schnellem Modell."""
        return self.generiere(
            [{"role": "user", "content": prompt}],
            modell_art="schnell",
            max_tokens=max_tokens,
        )

    def generiere_code(self, prompt: str, max_tokens: int = 1000) -> str | None:
        """Code-Generierung mit Code-Modell."""
        return self.generiere(
            [{"role": "user", "content": prompt}],
            modell_art="code",
            max_tokens=max_tokens,
        )

    # ─── Status ───────────────────────────────────────────────────────────────
    def status(self) -> dict:
        return {
            "verfuegbar": self.ist_verfuegbar(),
            "host": self._host,
            "modelle": self._modelle,
        }
