"""Nova Predator v3 — OllamaClient.
AsyncClient wrapper für Chat, Embed, Status.
Python 3.14: asyncio.run() / asyncio.get_running_loop() pattern.
Kein Nova-Import außer logger.

v2-Änderungen:
  - VRAM-Semaphore: verhindert gleichzeitiges Laden großer Modelle
  - geladene_modelle(): gibt aktuell im VRAM befindliche Modelle zurück
  - vram_warten(): wartet bis VRAM unter Zielgrenze fällt (für Handoff-Gate)

v3-Änderungen (additiv, kein bestehender Code verändert):
  - ToolAufruf: Dataclass für erkannte Tool-Calls
  - chat_mit_tools(): Prompt-basiertes Tool-Calling
  - stream_mit_tools(): Streaming-Variante mit Tool-Detection
"""
from __future__ import annotations
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
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


# ── Tool-Calling Datenstrukturen (v3) ────────────────────────────────────────

@dataclass
class ToolAufruf:
    """Repräsentiert einen vom Modell erkannten Tool-Call."""
    name: str
    argumente: dict[str, Any] = field(default_factory=dict)


def _tool_system_prompt(tools: list[dict]) -> str:
    """Baut den Tool-Abschnitt für den System-Prompt.

    Prompt-basiertes Tool-Calling: funktioniert mit jedem Modell,
    kein SDK-Feature erforderlich, bewährt im Codebase (react_brain.py).

    Das Modell antwortet mit:
        <tool_call>{"name": "wetter", "argumente": {"stadt": "Berlin"}}</tool_call>
    Oder normal ohne Tag wenn kein Tool gebraucht wird.
    """
    if not tools:
        return ""

    tool_beschreibungen: list[str] = []
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])

        param_texte: list[str] = []
        for pname, pinfo in params.items():
            pflicht = " (erforderlich)" if pname in required else " (optional)"
            param_texte.append(f"    - {pname}: {pinfo.get('description', '')}{pflicht}")

        param_block = "\n".join(param_texte) if param_texte else "    (keine Parameter)"
        tool_beschreibungen.append(f"• {name}: {desc}\n{param_block}")

    tools_text = "\n\n".join(tool_beschreibungen)
    return (
        f"Du hast Zugriff auf folgende Tools:\n\n{tools_text}\n\n"
        "Wenn du ein Tool nutzen möchtest, antworte NUR mit:\n"
        "<tool_call>{\"name\": \"tool_name\", \"argumente\": {...}}</tool_call>\n\n"
        "Wenn du kein Tool brauchst, antworte normal ohne diesen Tag.\n"
        "Nutze ein Tool nur wenn es die Anfrage direkt verbessert."
    )


def _strip_think_tags(text: str) -> str:
    """Entfernt qwen3 Thinking-Mode Tags (<think>...</think>) aus dem Output.

    Qwen3 gibt im Thinking-Mode seine interne Reasoning-Kette in <think>-Tags aus.
    Für den User-sichtbaren Chat-Output müssen die entfernt werden.
    """
    if "<think>" not in text and "</think>" not in text:
        return text
    # Kompletten <think>...</think> Block entfernen (mit DOTALL für Mehrzeiler)
    bereinigt = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    # Falls nur ein Tag (unfertig) vorhanden ist — den Rest nach </think> nehmen
    if "</think>" in bereinigt:
        bereinigt = bereinigt.split("</think>", 1)[1]
    # Falls <think> da ist ohne schließenden Tag — alles davor behalten
    if "<think>" in bereinigt:
        bereinigt = bereinigt.split("<think>", 1)[0]
    return bereinigt.strip()


def _parse_tool_aufrufe(text: str) -> tuple[list[ToolAufruf], str]:
    """Extrahiert Tool-Calls aus Modell-Antwort.

    Returns:
        (tool_aufrufe, bereinigter_text_ohne_tags)
    """
    # v3.1: Zuerst Think-Tags entfernen (qwen3 Thinking-Mode)
    text = _strip_think_tags(text)

    tool_aufrufe: list[ToolAufruf] = []
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    for match in pattern.finditer(text):
        roh = match.group(1).strip()
        try:
            daten = json.loads(roh)
            name = daten.get("name", "").strip()
            # "argumente" (Deutsch) und "arguments" (Englisch) beide akzeptieren
            argumente = daten.get("argumente", daten.get("arguments", {}))
            if name:
                tool_aufrufe.append(ToolAufruf(name=name, argumente=argumente))
                log.debug("Tool-Call erkannt: %s %s", name, argumente)
        except json.JSONDecodeError as e:
            log.warning("Tool-Call JSON-Parse Fehler: %s | roh: %s", e, roh[:100])

    bereinigt = pattern.sub("", text).strip()
    return tool_aufrufe, bereinigt


# ── OllamaClient ─────────────────────────────────────────────────────────────

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
        """Einfacher Chat-Call (kein Streaming). Für Brain-LLM.

        v3.3: Wenn optionen['think'] gesetzt ist, wird es als Ollama
        think-Parameter übergeben (Ollama ≥0.6, qwen3-Modelle).
        """
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(nachrichten)

        # think aus optionen extrahieren — ist ein Top-Level-Param, keine Option
        opts = dict(optionen) if optionen else {}
        think_flag = opts.pop("think", None)

        try:
            client = self._get_client()
            kwargs: dict[str, Any] = {
                "model":    modell,
                "messages": msgs,
                "options":  opts,
            }
            if think_flag is not None:
                kwargs["think"] = bool(think_flag)
            resp = await client.chat(**kwargs)
            # think:true → resp.message.thinking enthält Reasoning, .content die Antwort
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
        """Streaming Chat — yield token für token.

        v3.1: Filtert <think>...</think> Blöcke (qwen3 Thinking-Mode) aus dem Stream.
        """
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(nachrichten)

        # State für Think-Tag-Filter
        buffer = ""
        in_think = False

        try:
            client = self._get_client()
            async for teil in await client.chat(
                model=modell,
                messages=msgs,
                stream=True,
                options=optionen or {},
            ):
                token = teil.message.content
                if not token:
                    continue

                buffer += token

                # Think-Tag Anfang erkennen — bis dahin emittierten Teil raus
                if not in_think and "<think>" in buffer:
                    vor_think, rest = buffer.split("<think>", 1)
                    if vor_think:
                        yield vor_think
                    buffer = rest
                    in_think = True

                # Im Think-Block — Buffer weiter füllen bis </think>
                if in_think:
                    if "</think>" in buffer:
                        _, nach_think = buffer.split("</think>", 1)
                        buffer = nach_think.lstrip()
                        in_think = False
                    else:
                        # Noch kein Ende — weiter buffern, nichts yielden
                        continue

                # Normal-Modus — emittieren wenn Tag-Anfang nicht in Sicht
                # (vorsichtig: könnte noch ein halbes "<think" drin sein)
                if not in_think and "<" not in buffer[-10:]:
                    # Sicher kein Tag am Rand — alles rausschicken
                    if buffer:
                        yield buffer
                        buffer = ""

            # Stream-Ende: Rest-Buffer noch yielden (falls vorhanden)
            if not in_think and buffer:
                # Zur Sicherheit nochmal finale Think-Tags strippen
                yield _strip_think_tags(buffer)

        except ResponseError as e:
            log.error("Stream ResponseError (Modell=%s): %s", modell, e)
        except Exception as e:
            log.error("Stream Fehler (Modell=%s): %s", modell, e)

    # ── Tool-Calling (v3, additiv) ────────────────────────────────────────────

    async def chat_mit_tools(
        self,
        nachrichten: list[dict[str, Any]],
        modell: str,
        system: str,
        tools: list[dict],
        optionen: dict[str, Any] | None = None,
    ) -> tuple[str, list[ToolAufruf]]:
        """Chat-Call mit Tool-Support (prompt-basiert).

        Das Modell bekommt Tool-Beschreibungen im System-Prompt.
        Es antwortet entweder mit <tool_call>...</tool_call> oder normal.

        Args:
            nachrichten: Chat-History (ohne System)
            modell:      Ollama-Modell-Name
            system:      Basis-System-Prompt (Tool-Abschnitt wird angehängt)
            tools:       Tool-Definitionen von SkillMesh.alle_tool_definitionen()
            optionen:    Ollama-Optionen (temperature etc.)

        Returns:
            (antwort_text, tool_aufrufe)
            Bei Tool-Calls: tool_aufrufe gefüllt, antwort_text meist leer
            Bei normaler Antwort: tool_aufrufe leer, antwort_text gefüllt
        """
        tool_abschnitt = _tool_system_prompt(tools)
        voll_system = f"{system}\n\n{tool_abschnitt}" if tool_abschnitt else system

        antwort_roh = await self.chat(
            nachrichten=nachrichten,
            modell=modell,
            system=voll_system,
            optionen=optionen,
        )

        tool_aufrufe, antwort_text = _parse_tool_aufrufe(antwort_roh)

        if tool_aufrufe:
            log.debug(
                "chat_mit_tools: %d Tool-Call(s): %s",
                len(tool_aufrufe),
                [t.name for t in tool_aufrufe],
            )
        else:
            log.debug(
                "chat_mit_tools: keine Tool-Calls, normale Antwort (%d Zeichen)",
                len(antwort_text),
            )

        return antwort_text, tool_aufrufe

    async def stream_mit_tools(
        self,
        nachrichten: list[dict[str, Any]],
        modell: str,
        system: str,
        tools: list[dict],
        optionen: dict[str, Any] | None = None,
    ) -> AsyncIterator[str | list[ToolAufruf]]:
        """Streaming mit Tool-Support.

        Strategie:
          1. chat_mit_tools() für Tool-Detection (non-streaming)
          2. Wenn Tool-Calls → yield list[ToolAufruf] als erstes Item
          3. Nach Tool-Ausführung wird dieser Generator erneut aufgerufen
             (durch nova_ws Tool-Loop) — dann ohne Tools, echtes Streaming

        Aufrufer muss prüfen: isinstance(item, list) → Tool-Calls

        Yielded:
          - list[ToolAufruf]  wenn Tool-Calls erkannt (exakt einmal, zuerst)
          - str               Text-Tokens der Antwort
        """
        antwort_text, tool_aufrufe = await self.chat_mit_tools(
            nachrichten=nachrichten,
            modell=modell,
            system=system,
            tools=tools,
            optionen=optionen,
        )

        if tool_aufrufe:
            yield tool_aufrufe  # type: ignore[misc]
            # Eventueller Begleittext (meist leer bei Tool-Calls)
            for ch in antwort_text:
                yield ch
        else:
            # Keine Tools — Text chunk-weise yielden
            if antwort_text:
                chunk = 4
                for i in range(0, len(antwort_text), chunk):
                    yield antwort_text[i:i + chunk]
                    await asyncio.sleep(0)  # Event-Loop nicht blockieren

    async def modell_laden(self, modell: str) -> bool:
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
