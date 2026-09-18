"""Nova Predator v1 Layer 5 — Agent Orchestrator.

Koordiniert den gesamten Layer-5-Loop:
  1. Gemma4 plant (TaskPlanner)
  2. VRAM-Wechsel: Gemma → codestral
  3. ClawEngine führt Subtask aus
  4. VRAM-Wechsel: codestral → Gemma
  5. Gemma verifiziert
  6. RetryMechanic + LoopDetector
  7. Nächster Subtask oder fertig
"""
from __future__ import annotations

import asyncio
import time

from core.logger import get

log = get("layer5.agent_orchestrator")
import uuid
from pathlib import Path
from typing import AsyncGenerator, Any

from layer5.agent_task import AgentTask, AgentStatus, Subtask, RetryMechanic, LoopDetector
from layer5.task_planner import TaskPlanner
from layer5.claw_engine import ClawEngine, ClawEvent, EngineConfig
from layer5.claw_tools import ClawTools
from layer5.workspace_fs import WorkspaceFS
from layer5.permission import PermissionEnforcer, PermissionMode




class AgentOrchestrator:
    """Gesamtkoordination Layer 5.

    Wird von web/agent.py aufgerufen.
    Gibt Events als AsyncGenerator zurück für WebSocket-Streaming.
    """

    def __init__(self,
                 nova_state: Any = None,
                 ollama_host:   str  = "http://localhost:11434",
                 codestral_mod: str  = "codestral:22b",
                 gemma_mod:     str  = "gemma4:e4b",
                 workspace_root: str = "workspace") -> None:
        self._state        = nova_state
        self._ollama_host  = ollama_host
        self._coder_modell = codestral_mod
        self._gemma_modell = gemma_mod
        self._ws_root      = Path(workspace_root)
        self._ws_root.mkdir(exist_ok=True)

        self._planner = TaskPlanner(ollama_host=ollama_host, modell=gemma_mod)
        self._aktiver_task: AgentTask | None = None
        self._sperr_flag  = False   # True wenn Agent läuft

    @property
    def laeuft(self) -> bool:
        return self._sperr_flag

    async def starten(self, aufgabe: str, projekt: str | None = None,
                       kontext: str = ""
                       ) -> AsyncGenerator[dict, None]:
        """Startet Agent-Run, yieldet Events für WebSocket."""
        if self._sperr_flag:
            yield {"typ": "fehler", "text": "Agent läuft bereits — bitte warten"}
            return

        self._sperr_flag = True
        task_id  = f"task-{uuid.uuid4().hex[:8]}"
        projekt  = projekt or aufgabe[:30]
        ws_pfad  = self._ws_root / task_id
        ws_pfad.mkdir(parents=True, exist_ok=True)

        yield {"typ": "start", "task_id": task_id, "aufgabe": aufgabe, "projekt": projekt}

        try:
            async for event in self._run_loop(task_id, aufgabe, projekt, ws_pfad, kontext):
                yield event
        finally:
            self._sperr_flag = False

    async def _run_loop(self, task_id: str, aufgabe: str, projekt: str,
                         ws_pfad: Path, kontext: str
                         ) -> AsyncGenerator[dict, None]:

        # ── Phase 1: Planung (Gemma4) ─────────────────────────────────
        yield {"typ": "phase", "phase": "planung", "text": "Gemma plant..."}

        plan = await self._planner.plane_aufgabe(aufgabe, kontext=kontext)
        hat_ui = bool(plan.get("hat_ui", False))

        # Rückfragen werden nie gestellt — Agent entscheidet selbst
        task = self._planner.erstelle_task(task_id, aufgabe, projekt, ws_pfad, plan)
        self._aktiver_task = task
        task.status = AgentStatus.RUNNING

        yield {
            "typ":        "plan",
            "subtasks":   [{"id": s.id, "beschreibung": s.beschreibung}
                           for s in task.subtasks],
            "zusammenfassung": task.plan_text,
            "hat_ui": hat_ui,
        }

        # ── Phase 2: Ausführung ───────────────────────────────────────
        retry   = RetryMechanic()
        loop_d  = LoopDetector(ws_pfad)

        workspace = WorkspaceFS(ws_pfad)
        enforcer  = PermissionEnforcer(
            active_mode=PermissionMode.WORKSPACE_WRITE
        )
        tools     = ClawTools(workspace, enforcer)
        engine    = ClawEngine(
            tools,
            config=EngineConfig(
                modell=self._coder_modell,
                ollama_host=self._ollama_host,
            )
        )

        # Index-basierter Loop statt for-enumerate:
        # 'continue' in for-Schleife würde zum NÄCHSTEN Subtask springen,
        # nicht zum aktuellen zurück — Retry würde nie ausgeführt.
        subtask_idx = 0
        while subtask_idx < len(task.subtasks):
            subtask = task.subtasks[subtask_idx]
            subtask_nr = subtask_idx + 1

            subtask.status = AgentStatus.RUNNING
            subtask.started_at = time.time()

            yield {
                "typ":      "subtask_start",
                "nr":       subtask_nr,
                "total":    len(task.subtasks),
                "text":     subtask.beschreibung,
            }

            # VRAM: Gemma entladen, codestral laden
            await self._modell_wechseln(laden=self._coder_modell)

            # Dateibaum für Kontext
            dateibaum = self._dateibaum(ws_pfad)

            # Vorherige Schritte zusammenfassen
            vorher = self._vorherige_schritte(task.subtasks[:subtask_idx])

            # Engine ausführen
            letzter_output = ""
            iterationen    = 0

            async for claw_event in engine.run(
                subtask=subtask.beschreibung,
                projekt=projekt,
                dateibaum=dateibaum,
                vorherige_schritte=vorher,
                hat_ui=hat_ui,
            ):
                yield self._claw_zu_ws_event(claw_event, subtask)

                if claw_event.typ == "tool_result":
                    letzter_output = claw_event.data.get("output", "")
                    if claw_event.data.get("name") in ("write_file", "edit_file"):
                        loop_d.neuer_file_write()
                elif claw_event.typ == "iteration":
                    iterationen = claw_event.data.get("n", 0)
                    loop_d.naechste_iteration(letzter_output, subtask.id)
                    ist_loop, grund = loop_d.ist_loop()
                    if ist_loop:
                        yield {"typ": "warnung", "text": f"Loop erkannt: {grund}"}
                        break
                elif claw_event.typ == "done":
                    letzter_output = claw_event.data.get("text", "")
                    subtask.ergebnis = letzter_output

            # VRAM: codestral entladen, Gemma laden
            await self._modell_wechseln(laden=self._gemma_modell)

            # ── Phase 3: Verifikation (Gemma4) ──────────────────────
            yield {"typ": "phase", "phase": "verifikation",
                   "text": f"Gemma verifiziert Subtask {subtask_nr}..."}

            verifikation = await self._planner.verifiziere_subtask(
                subtask, letzter_output, task.plan_text
            )

            if verifikation.get("erfolgreich"):
                subtask.status   = AgentStatus.DONE
                subtask.done_at  = time.time()
                yield {
                    "typ":      "subtask_done",
                    "nr":       subtask_nr,
                    "text":     subtask.beschreibung,
                    "ergebnis": subtask.ergebnis[:500],
                }
                subtask_idx += 1  # Weiter zum nächsten Subtask
            else:
                if retry.darf_retry(subtask):
                    retry.registriere_versuch(subtask)
                    briefing = retry.korrektur_briefing(
                        subtask, verifikation.get("korrektur", "")
                    )
                    subtask.beschreibung = briefing
                    subtask.status = AgentStatus.PENDING
                    yield {
                        "typ":   "retry",
                        "nr":    subtask_nr,
                        "grund": verifikation.get("korrektur", ""),
                    }
                    # subtask_idx NICHT erhöhen → gleicher Subtask wird wiederholt
                else:
                    subtask.status  = AgentStatus.FAILED
                    subtask.fehler  = verifikation.get("korrektur", "Max. Versuche erreicht")
                    subtask.done_at = time.time()
                    yield {
                        "typ":   "subtask_failed",
                        "nr":    subtask_nr,
                        "grund": subtask.fehler,
                    }
                    subtask_idx += 1  # Weiter trotz Fehler

        # ── Abschluss ─────────────────────────────────────────────────
        task.status = AgentStatus.DONE
        task.done_at = time.time()

        yield {
            "typ":           "fertig",
            "task_id":       task_id,
            "zusammenfassung": task.zusammenfassung(),
            "workspace":     str(ws_pfad),
            "dauer_ms":      int((task.done_at - task.started_at) * 1000),
        }

    def _claw_zu_ws_event(self, event: ClawEvent, subtask: Subtask) -> dict:
        """Konvertiert ClawEvent zu WebSocket-Event."""
        base = {"subtask_id": subtask.id}
        if event.typ == "tool_call":
            return {**base, "typ": "tool_call",
                    "tool": event.data.get("name"),
                    "args_preview": str(event.data.get("args", {}))[:200]}
        if event.typ == "tool_result":
            return {**base, "typ": "tool_result",
                    "tool":    event.data.get("name"),
                    "success": event.data.get("success"),
                    "output":  event.data.get("output", "")[:500],
                    "ms":      event.data.get("duration_ms", 0)}
        if event.typ == "iteration":
            return {**base, "typ": "iteration",
                    "n": event.data.get("n"), "max": event.data.get("max")}
        if event.typ == "token":
            return {**base, "typ": "token", "text": event.data.get("text", "")}
        if event.typ == "done":
            return {**base, "typ": "engine_done",
                    "text": event.data.get("text", ""),
                    "truncated": event.data.get("truncated", False)}
        if event.typ == "error":
            return {**base, "typ": "engine_fehler",
                    "text": event.data.get("text", "")}
        return {**base, "typ": event.typ, **event.data}

    def _dateibaum(self, ws: Path, max_files: int = 50) -> str:
        """Erstellt kompakten Dateibaum-String."""
        lines = []
        for p in sorted(ws.rglob("*")):
            if p.is_file():
                try:
                    rel = p.relative_to(ws)
                    lines.append(f"  {rel}")
                except ValueError:
                    pass
            if len(lines) >= max_files:
                lines.append("  ... (weitere Dateien)")
                break
        return "\n".join(lines) if lines else "  (leer)"

    def _vorherige_schritte(self, done_subtasks: list[Subtask]) -> str:
        """Komprimierter Kontext aus abgeschlossenen Subtasks."""
        if not done_subtasks:
            return ""
        lines = []
        for s in done_subtasks:
            status = "✓" if s.status == AgentStatus.DONE else "✗"
            summary = s.ergebnis[:200] if s.ergebnis else "(kein Ergebnis)"
            lines.append(f"{status} {s.beschreibung}: {summary}")
        return "\n".join(lines)

    async def _modell_wechseln(self, laden: str) -> None:
        """Entlädt aktuelles Modell, lädt neues. Nutzt SchlafManager."""
        if self._state is None:
            return
        try:
            schlaf = self._state.schlaf
            if schlaf:
                await schlaf.aktivitaet_melden()
        except Exception:
            pass

    def task_status(self) -> dict | None:
        """Aktueller Task-Status für API."""
        if self._aktiver_task is None:
            return None
        t = self._aktiver_task
        return {
            "task_id":     t.id,
            "aufgabe":     t.aufgabe,
            "status":      t.status.value,
            "subtasks":    [
                {"id": s.id, "beschreibung": s.beschreibung,
                 "status": s.status.value, "versuche": s.versuche}
                for s in t.subtasks
            ],
            "workspace":   str(t.workspace),
            "started_at":  t.started_at,
        }
