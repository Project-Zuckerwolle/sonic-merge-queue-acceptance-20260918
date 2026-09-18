"""Nova Predator v1 Layer 6 — Apex Orchestrator.

Koordiniert:
  - ReActBrain (Kern-Loop)
  - MemoryBridge (Brain + Session Integration)
  - ComputerController (App-Steuerung)
  - Task-Verwaltung + WebSocket-Streaming

Wird von web/apex.py (WebSocket) aufgerufen.
Singleton-Pattern wie web/agent.py (Layer 5).
"""
from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Any, TYPE_CHECKING

from core.logger import get
from layer6.apex_types import ApexTask, ApexStatus, ReActStep, ReActTyp, TraceLog
from layer6.react_brain import ReActBrain
from layer6.memory_bridge import MemoryBridge

if TYPE_CHECKING:
    from core.ollama_client import OllamaClient
    from core.brain_manager import BrainManager
    from memory.session import Session

log = get("apex.orchestrator")

_WORKSPACE_ROOT = Path("workspace_apex")


class ApexOrchestrator:
    """Haupt-Koordinator für Layer 6.

    Singleton — eine Instanz für die gesamte Nova-Laufzeit.
    """

    def __init__(
        self,
        ollama: "OllamaClient",
        brain_manager: "BrainManager",
        session: "Session",
        apex_modell:  str = "qwen3:30b-a3b",
        coder_modell: str = "qwen3-coder:30b",
        workspace_root: str | Path = "workspace_apex",
    ) -> None:
        self._ollama       = ollama
        self._brain        = brain_manager
        self._session      = session
        self._apex_modell  = apex_modell
        self._coder_modell = coder_modell
        self._ws_root      = Path(workspace_root)
        self._ws_root.mkdir(parents=True, exist_ok=True)

        self._brain_loop   = ReActBrain(
            ollama,
            chat_modell=apex_modell,
            coder_modell=coder_modell,
        )
        self._memory_bridge = MemoryBridge(brain_manager, session)

        self._aktiver_task: ApexTask | None = None
        self._laeuft = False

        # Computer-Controller (optional, Windows-only)
        self._computer: "ComputerController | None" = None
        try:
            from layer6.computer_controller import ComputerController
            self._computer = ComputerController()
        except Exception:
            log.debug("ComputerController nicht verfügbar (kein Windows oder fehlende Pakete)")

    @property
    def laeuft(self) -> bool:
        return self._laeuft

    def task_status(self) -> dict | None:
        if self._aktiver_task is None:
            return None
        t = self._aktiver_task
        return {
            "task_id":    t.id,
            "aufgabe":    t.aufgabe,
            "status":     t.status.value,
            "iteration":  t.iteration,
            "max_iter":   t.max_iter,
            "schritte":   len(t.schritte),
            "workspace":  str(t.workspace),
        }

    # Keywords die auf Coding-Task hindeuten
    _CODE_KEYWORDS = frozenset({
        "code", "script", "programm", "python", "javascript", "html", "css",
        "baue", "erstelle", "schreibe", "implementiere", "entwickle", "debug",
        "fix", "repariere", "funktion", "klasse", "api", "app", "tool",
        "gui", "ui", "fenster", "oberfläche", "skript",
    })

    @staticmethod
    def _erkennt_coding(aufgabe: str) -> bool:
        """Erkennt ob eine Aufgabe Coding erfordert."""
        text = aufgabe.lower()
        return any(kw in text for kw in ApexOrchestrator._CODE_KEYWORDS)

    async def starten(
        self,
        aufgabe: str,
        max_iter: int = 40,
        kontext: str = "",
    ) -> AsyncGenerator[dict, None]:
        """Startet einen neuen Apex-Task. Yieldet WebSocket-Events."""
        if self._laeuft:
            yield {"typ": "fehler", "text": "Apex läuft bereits — bitte warten"}
            return

        self._laeuft = True
        task_id  = f"apex-{uuid.uuid4().hex[:8]}"
        ws_pfad  = self._ws_root / task_id
        ws_pfad.mkdir(parents=True, exist_ok=True)

        vollstaendige_aufgabe = aufgabe
        if kontext:
            vollstaendige_aufgabe += f"\n\nZusatzkontext: {kontext}"

        hat_code = self._erkennt_coding(vollstaendige_aufgabe)

        task = ApexTask(
            id=task_id,
            aufgabe=vollstaendige_aufgabe,
            max_iter=max_iter,
            workspace=ws_pfad,
        )
        self._aktiver_task = task

        yield {
            "typ": "start",
            "task_id": task_id,
            "aufgabe": aufgabe,
            "modell": self._coder_modell if hat_code else self._apex_modell,
        }

        try:
            async for event in self._run_loop(task, hat_code=hat_code):
                yield event
        except Exception as e:
            log.error("Apex-Orchestrator Fehler: %s", e)
            yield {"typ": "fehler", "text": str(e)}
        finally:
            self._laeuft = False

    async def _run_loop(self, task: ApexTask,
                         hat_code: bool = False) -> AsyncGenerator[dict, None]:
        """Führt Task durch alle Phasen."""

        # Extra-Tools: Computer-Controller wenn verfügbar
        extra_tools: dict[str, Any] = {}
        if self._computer:
            extra_tools.update({
                "app_starten":   self._computer.app_starten,
                "app_beenden":   self._computer.app_beenden,
                "prozesse":      self._computer.prozesse_liste,
                "volume":        self._computer.volume_setzen,
                "screenshot":    self._computer.screenshot,
                "clipboard":     self._computer.clipboard_setzen,
            })

        # ReAct-Loop starten
        async for event in self._brain_loop.run(task, extra_tools=extra_tools, hat_code=hat_code):
            # Events weiterleiten + bei Reflexion in Memory-Bridge
            yield event

            # Reflexions-Fakten laufend in SessionFacts eintragen
            if event.get("typ") == "reflexion" and event.get("fakten"):
                for fakt in event["fakten"][:2]:
                    await self._memory_bridge.fortschritts_eintrag(
                        task.id, task.aufgabe, fakt
                    )

            # Bei Fortschritt: Dateien im Workspace registrieren
            if event.get("typ") == "observation" and event.get("success"):
                dateien = self._workspace_dateien(task.workspace)
                if dateien:
                    yield {"typ": "workspace_update", "dateien": dateien}

        # Nach Abschluss: Memory-Bridge
        if task.status in (ApexStatus.DONE, ApexStatus.FAILED):
            stats = await self._memory_bridge.nach_task(task)
            yield {
                "typ":    "memory_sync",
                "brain":  stats.get("brain", 0),
                "session_facts": stats.get("session_facts", 0),
            }

    def _workspace_dateien(self, ws: Path) -> list[dict]:
        """Gibt Dateiliste im Workspace zurück."""
        dateien = []
        for p in sorted(ws.iterdir()):
            if p.is_file() and not p.name.endswith(".jsonl"):
                dateien.append({"name": p.name, "groesse": p.stat().st_size})
        return dateien[:20]

    def abbrechen(self) -> None:
        """Bricht laufenden Task ab."""
        if self._aktiver_task:
            self._aktiver_task.status = ApexStatus.ABORTED
        self._laeuft = False
