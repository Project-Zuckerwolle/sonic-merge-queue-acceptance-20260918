"""Nova Predator v1 Layer 5 — Claw Engine (Agentic Loop).

Implementiert Claws ConversationRuntime in Python + Ollama.
Der Loop:
  1. System-Prompt + Tool-Definitionen an Modell schicken
  2. Tool-Calls parsen
  3. Tools ausführen (ClawTools)
  4. Ergebnis an Modell zurückgeben
  5. Wiederholen bis "fertig" oder max_iterations erreicht

Modell: codestral:22b via Ollama OpenAI-compat Layer
"""
from __future__ import annotations

import asyncio

from core.logger import get

log = get("layer5.claw_engine")
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from layer5.claw_tools import ClawTools, ToolResult
from layer5.workspace_fs import WorkspaceFS
from layer5.permission import PermissionEnforcer, PermissionMode, TOOL_PERMISSIONS


@dataclass
class ClawEvent:
    """Event das der Engine-Loop nach außen emittiert."""
    typ:  str   # "tool_call", "tool_result", "token", "done", "error", "iteration"
    data: dict  = field(default_factory=dict)


@dataclass
class EngineConfig:
    modell:         str   = "codestral:22b"
    ollama_host:    str   = "http://localhost:11434"
    max_iterations: int   = 20
    temperature:    float = 0.1
    max_tokens:     int   = 8192


# Tool-Definitionen im OpenAI-Format (aus Claws mvp_tool_specs)
TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Liest eine Textdatei aus dem Workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":   {"type": "string", "description": "Dateipfad"},
                    "offset": {"type": "integer", "description": "Start-Zeile (0-basiert)"},
                    "limit":  {"type": "integer", "description": "Anzahl Zeilen"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Schreibt eine Datei in den Workspace (erstellt Verzeichnisse automatisch).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Ersetzt Text in einer bestehenden Datei.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":        {"type": "string"},
                    "old_string":  {"type": "string"},
                    "new_string":  {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Führt Shell-Befehle im Workspace aus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob_search",
            "description": "Findet Dateien per Glob-Pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path":    {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Sucht Datei-Inhalte per Regex.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path":    {"type": "string"},
                    "glob":    {"type": "string"},
                    "-i":      {"type": "boolean"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "REPL",
            "description": "Führt Code in einem REPL aus (python, javascript).",
            "parameters": {
                "type": "object",
                "properties": {
                    "code":       {"type": "string"},
                    "language":   {"type": "string"},
                    "timeout_ms": {"type": "integer"},
                },
                "required": ["code", "language"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "WebFetch",
            "description": "Lädt eine URL und gibt den Inhalt zurück.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url":    {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["url", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "WebSearch",
            "description": "Sucht im Web nach aktuellen Informationen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "SendUserMessage",
            "description": "Sendet eine Nachricht an den User (Fortschritt, Fragen).",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "status":  {"type": "string", "enum": ["normal", "proactive"]},
                },
                "required": ["message", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "design_ui",
            "description": (
                "Erstellt eine vollständige UI-Datei. Nutze dieses Tool wenn "
                "eine GUI, Oberfläche, App oder visuelles Programm gebaut werden soll. "
                "Wähle automatisch das richtige Framework: "
                "tkinter für Desktop-Apps (kein Install nötig), "
                "HTML/CSS/JS für Web-Apps, "
                "PyQt/customtkinter wenn explizit gewünscht."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "framework": {
                        "type": "string",
                        "enum": ["tkinter", "html", "customtkinter", "pyqt5"],
                        "description": "UI-Framework",
                    },
                    "beschreibung": {
                        "type": "string",
                        "description": "Was die UI zeigen/können soll",
                    },
                    "ausgabe_datei": {
                        "type": "string",
                        "description": "Dateiname z.B. 'app.py' oder 'index.html'",
                    },
                    "stil": {
                        "type": "string",
                        "enum": ["modern-dark", "modern-light", "minimal", "classic"],
                        "description": "Visueller Stil",
                    },
                    "komponenten": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "UI-Elemente z.B. ['button:Sortieren', 'label:Pfad', 'listbox:Dateien', 'progressbar']",
                    },
                },
                "required": ["framework", "beschreibung", "ausgabe_datei"],
            },
        },
    },
]


def _build_system_prompt(subtask_beschreibung: str, projekt: str,
                          dateibaum: str, vorherige_schritte: str,
                          hat_ui: bool = False) -> str:
    """Baut den System-Prompt für codestral:22b."""
    ui_hinweis = ""
    if hat_ui:
        ui_hinweis = """
WICHTIG — DIESE AUFGABE HAT EINE UI:
- Nutze design_ui für alle Fenster/Oberflächen
- tkinter ist bevorzugt (kein Install, in Python stdlib)
- Baue eine saubere, moderne UI mit klarem Layout
- Trenne Logik (logic.py) von UI (app.py)
"""
    return f"""Du bist ein präziser Coding-Agent. Du implementierst Code direkt — keine Erklärungen, keine Fragen.
{ui_hinweis}
PROJEKT: {projekt}
AKTUELLE AUFGABE: {subtask_beschreibung}

WORKSPACE-DATEIEN:
{dateibaum}

VORHERIGE SCHRITTE:
{vorherige_schritte if vorherige_schritte else "Erster Schritt."}

REGELN:
- Implementiere sofort, entscheide selbst bei unklaren Details
- Nutze vernünftige Defaults (UTF-8, aufsteigend, YAML-Config etc.)
- Nach dem Schreiben: teste mit REPL oder bash
- SendUserMessage nur bei wichtigen Meilensteinen, nicht für Statusupdates
- Antworte am Ende mit einer Zeile was fertig ist
- Alle Dateien bleiben im Workspace
"""


class ClawEngine:
    """Der Agentic Loop — Claws ConversationRuntime in Python.

    Wird von AgentOrchestrator pro Subtask aufgerufen.
    Gibt Events als AsyncGenerator zurück.
    """

    def __init__(self, tools: ClawTools, config: EngineConfig | None = None) -> None:
        self.tools  = tools
        self.cfg    = config or EngineConfig()
        self._history: list[dict] = []

    async def run(
        self,
        subtask:          str,
        projekt:          str,
        dateibaum:        str = "",
        vorherige_schritte: str = "",
        context_snapshot: str = "",
        hat_ui:           bool = False,
    ) -> AsyncGenerator[ClawEvent, None]:
        """Hauptloop — führt Subtask aus, yieldet Events."""
        self._history = []
        system_prompt = _build_system_prompt(
            subtask, projekt, dateibaum, vorherige_schritte, hat_ui=hat_ui
        )

        user_msg = subtask
        if context_snapshot:
            user_msg = f"{subtask}\n\nKontext aus vorherigen Schritten:\n{context_snapshot}"

        self._history.append({"role": "user", "content": user_msg})

        for iteration in range(1, self.cfg.max_iterations + 1):
            yield ClawEvent("iteration", {"n": iteration, "max": self.cfg.max_iterations})

            # LLM aufrufen
            response = await self._llm_call(system_prompt)
            if response is None:
                yield ClawEvent("error", {"text": "LLM nicht erreichbar"})
                return

            # Tool-Calls extrahieren
            tool_calls = response.get("tool_calls", [])
            content    = response.get("content", "")

            # Antwort zur History
            self._history.append({
                "role":       "assistant",
                "content":    content or "",
                "tool_calls": tool_calls,
            })

            # Wenn Text aber keine Tool-Calls → fertig
            if not tool_calls:
                if content:
                    yield ClawEvent("done", {"text": content, "iterations": iteration})
                    return
                # Leere Antwort → Fehler
                yield ClawEvent("error", {"text": "Leere LLM-Antwort ohne Tool-Calls"})
                return

            # Tool-Calls ausführen
            tool_results: list[dict] = []
            for tc in tool_calls:
                fn   = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                yield ClawEvent("tool_call", {"name": name, "args": args})

                result: ToolResult = await asyncio.get_running_loop().run_in_executor(
                    None, self.tools.execute, name, args
                )

                yield ClawEvent("tool_result", {
                    "name":     name,
                    "success":  result.success,
                    "output":   result.output[:2000] if result.output else result.error,
                    "duration_ms": result.duration_ms,
                })

                tool_results.append({
                    "tool_call_id": tc.get("id", f"call_{name}"),
                    "role":         "tool",
                    "name":         name,
                    "content":      result.output if result.success else f"ERROR: {result.error}",
                })

            # Tool-Ergebnisse zur History
            self._history.extend(tool_results)

        # Max iterations erreicht
        yield ClawEvent("done", {
            "text":       "Maximum Iterationen erreicht",
            "iterations": self.cfg.max_iterations,
            "truncated":  True,
        })

    async def _llm_call(self, system_prompt: str) -> dict | None:
        """Ruft Ollama über OpenAI-compat API auf."""
        import urllib.request, urllib.error

        url  = f"{self.cfg.ollama_host}/v1/chat/completions"
        body = {
            "model":       self.cfg.modell,
            "messages":    [{"role": "system", "content": system_prompt}] + self._history,
            "tools":       TOOL_DEFS,
            "tool_choice": "auto",
            "temperature": self.cfg.temperature,
            "max_tokens":  self.cfg.max_tokens,
        }

        try:
            data = json.dumps(body).encode("utf-8")
            req  = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            choice  = result.get("choices", [{}])[0]
            message = choice.get("message", {})
            return {
                "content":    message.get("content", ""),
                "tool_calls": message.get("tool_calls", []),
            }
        except urllib.error.URLError as e:
            return None
        except (json.JSONDecodeError, KeyError, IndexError):
            return None

    @property
    def history(self) -> list[dict]:
        return self._history
