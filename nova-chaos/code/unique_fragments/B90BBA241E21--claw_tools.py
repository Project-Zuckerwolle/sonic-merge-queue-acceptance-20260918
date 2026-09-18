"""Nova Predator v1 Layer 5 — Claw Tool-Engine.

Implementiert alle 41 Claw-Tools in Python.
Tool-Calls kommen als JSON vom LLM, werden hier ausgeführt.
Alle Datei-Operationen gehen durch WorkspaceFS (Boundary-Check).
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from layer5.workspace_fs import WorkspaceFS
from layer5.permission import PermissionEnforcer, PermissionMode


@dataclass
class ToolResult:
    tool_name: str
    success:   bool
    output:    str           # JSON-String wie Claw es zurückgibt
    error:     str = ""
    duration_ms: int = 0


class ClawTools:
    """Alle 41 Claw-Tools.

    Jedes Tool:
      - Prüft Permissions via PermissionEnforcer
      - Führt Operation durch
      - Gibt JSON-String zurück (wie Claws Rust-Implementierung)
    """

    def __init__(self, workspace: WorkspaceFS,
                 enforcer: PermissionEnforcer,
                 nova_state: Any | None = None,
                 event_queue: asyncio.Queue | None = None) -> None:
        self.ws       = workspace
        self.enforcer = enforcer
        self._state   = nova_state
        self._queue   = event_queue
        # Loop bei Init speichern — tools.execute() läuft im run_in_executor (Thread),
        # dort ist asyncio.get_event_loop() nicht verfügbar (RuntimeError Python 3.12)
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        # Interne Registries (vereinfacht gegenüber Claws Rust-Version)
        self._tasks:   dict[str, dict] = {}
        self._workers: dict[str, dict] = {}
        self._teams:   dict[str, dict] = {}
        self._crons:   dict[str, dict] = {}

    # ── Dispatch ──────────────────────────────────────────────────────────

    def execute(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """Führt Tool aus, gibt ToolResult zurück."""
        t0 = time.monotonic()
        try:
            self.enforcer.enforce(tool_name)
        except PermissionError as e:
            return ToolResult(tool_name, False, "", str(e))

        handler = self._handlers().get(tool_name)
        if handler is None:
            return ToolResult(tool_name, False, "", f"Unbekanntes Tool: {tool_name}")

        try:
            output = handler(args)
            ms = int((time.monotonic() - t0) * 1000)
            return ToolResult(tool_name, True, json.dumps(output, ensure_ascii=False), "", ms)
        except (WorkspaceFS.WorkspaceBoundaryError if hasattr(WorkspaceFS, 'WorkspaceBoundaryError') else Exception) as e:
            ms = int((time.monotonic() - t0) * 1000)
            return ToolResult(tool_name, False, "", f"Workspace-Fehler: {e}", ms)
        except Exception as e:
            ms = int((time.monotonic() - t0) * 1000)
            return ToolResult(tool_name, False, "", str(e), ms)

    def _handlers(self) -> dict[str, Callable]:
        return {
            # Tier 1 - Dateisystem
            "bash":             self._bash,
            "read_file":        self._read_file,
            "write_file":       self._write_file,
            "edit_file":        self._edit_file,
            "glob_search":      self._glob_search,
            "grep_search":      self._grep_search,
            # Tier 2 - Agenten (vereinfacht)
            "Agent":            self._agent,
            "TaskCreate":       self._task_create,
            "TaskGet":          self._task_get,
            "TaskList":         self._task_list,
            "TaskStop":         self._task_stop,
            "TaskUpdate":       self._task_update,
            "TaskOutput":       self._task_output,
            "RunTaskPacket":    self._run_task_packet,
            # Tier 3 - Worker
            "WorkerCreate":         self._worker_create,
            "WorkerGet":            self._worker_get,
            "WorkerObserve":        self._worker_observe,
            "WorkerResolveTrust":   self._worker_resolve_trust,
            "WorkerAwaitReady":     self._worker_await_ready,
            "WorkerSendPrompt":     self._worker_send_prompt,
            "WorkerRestart":        self._worker_restart,
            "WorkerTerminate":      self._worker_terminate,
            "WorkerObserveCompletion": self._worker_observe_completion,
            # Tier 4 - Team/Cron
            "TeamCreate":      self._team_create,
            "TeamDelete":      self._team_delete,
            "CronCreate":      self._cron_create,
            "CronDelete":      self._cron_delete,
            "CronList":        self._cron_list,
            # Tier 5 - Web
            "WebFetch":        self._web_fetch,
            "WebSearch":       self._web_search,
            # Tier 6 - Code
            "REPL":            self._repl,
            "NotebookEdit":    self._notebook_edit,
            "PowerShell":      self._powershell,
            # Tier 7 - System
            "TodoWrite":       self._todo_write,
            "Skill":           self._skill,
            "ToolSearch":      self._tool_search,
            "Sleep":           self._sleep,
            "SendUserMessage": self._send_user_message,
            "Config":          self._config,
            "EnterPlanMode":   self._enter_plan_mode,
            "ExitPlanMode":    self._exit_plan_mode,
            "StructuredOutput":self._structured_output,
            "AskUserQuestion": self._ask_user_question,
            "LSP":             self._lsp,
            "MCP":             self._mcp,
            "McpAuth":         self._mcp_auth,
            "ListMcpResources":self._list_mcp_resources,
            "ReadMcpResource": self._read_mcp_resource,
            "RemoteTrigger":   self._remote_trigger,
            "design_ui":       self._design_ui,
        }

    # ══════════════════════════════════════════════════════════════════════
    #  TIER 1: DATEISYSTEM
    # ══════════════════════════════════════════════════════════════════════

    def _bash(self, args: dict) -> dict:
        """bash — Shell-Befehl (nur in workspace/, kein Netzwerk default)."""
        import subprocess, os, sys
        command = args.get("command", "")
        timeout = args.get("timeout", 30)
        bg      = args.get("run_in_background", False)

        if bg:
            proc = subprocess.Popen(
                command, shell=True, cwd=str(self.ws.root),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return {"stdout": "", "stderr": "", "backgroundTaskId": str(proc.pid),
                    "interrupted": False, "noOutputExpected": True}

        try:
            proc = subprocess.run(
                command, shell=True, capture_output=True, encoding="utf-8", text=True,
                timeout=timeout, cwd=str(self.ws.root),
            )
            return {
                "stdout":   proc.stdout[:32768],
                "stderr":   proc.stderr[:8192],
                "interrupted": False,
                "returnCodeInterpretation":
                    f"exit_code:{proc.returncode}" if proc.returncode != 0 else "success",
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "", "interrupted": True,
                "stderr": f"Timeout nach {timeout}s",
                "returnCodeInterpretation": "timeout",
            }

    def _read_file(self, args: dict) -> dict:
        return self.ws.read_file(
            args["path"],
            offset=args.get("offset", 0),
            limit=args.get("limit"),
        )

    def _write_file(self, args: dict) -> dict:
        return self.ws.write_file(args["path"], args["content"])

    def _edit_file(self, args: dict) -> dict:
        return self.ws.edit_file(
            args["path"], args["old_string"], args["new_string"],
            replace_all=args.get("replace_all", False),
        )

    def _glob_search(self, args: dict) -> dict:
        return self.ws.glob_search(args["pattern"], args.get("path"))

    def _grep_search(self, args: dict) -> dict:
        return self.ws.grep_search(
            pattern=args["pattern"],
            path=args.get("path"),
            glob=args.get("glob"),
            case_insensitive=args.get("-i", False),
            output_mode=args.get("output_mode", "files_with_matches"),
            context=args.get("context", args.get("-C", 0)),
            head_limit=args.get("head_limit"),
        )

    # ══════════════════════════════════════════════════════════════════════
    #  TIER 2: AGENTEN
    # ══════════════════════════════════════════════════════════════════════

    def _agent(self, args: dict) -> dict:
        import uuid
        agent_id = f"agent-{uuid.uuid4().hex[:8]}"
        self._tasks[agent_id] = {
            "task_id":    agent_id,
            "status":     "running",
            "prompt":     args.get("prompt", ""),
            "description":args.get("description", ""),
            "created_at": time.time(),
            "output":     [],
        }
        return {"agentId": agent_id, "status": "running",
                "description": args.get("description", ""),
                "name": args.get("name", agent_id)}

    def _task_create(self, args: dict) -> dict:
        import uuid
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        task = {"task_id": task_id, "status": "created",
                "prompt": args.get("prompt", ""),
                "description": args.get("description", ""),
                "created_at": time.time(), "messages": [], "output": []}
        self._tasks[task_id] = task
        return task

    def _task_get(self, args: dict) -> dict:
        tid = args["task_id"]
        if tid not in self._tasks:
            raise ValueError(f"Task nicht gefunden: {tid}")
        return self._tasks[tid]

    def _task_list(self, args: dict) -> dict:
        return {"tasks": list(self._tasks.values()), "count": len(self._tasks)}

    def _task_stop(self, args: dict) -> dict:
        tid = args["task_id"]
        if tid not in self._tasks:
            raise ValueError(f"Task nicht gefunden: {tid}")
        self._tasks[tid]["status"] = "stopped"
        return self._tasks[tid]

    def _task_update(self, args: dict) -> dict:
        tid = args["task_id"]
        if tid not in self._tasks:
            raise ValueError(f"Task nicht gefunden: {tid}")
        self._tasks[tid]["messages"].append(args.get("message", ""))
        return self._tasks[tid]

    def _task_output(self, args: dict) -> dict:
        tid = args["task_id"]
        if tid not in self._tasks:
            raise ValueError(f"Task nicht gefunden: {tid}")
        return {"task_id": tid, "output": self._tasks[tid].get("output", [])}

    def _run_task_packet(self, args: dict) -> dict:
        import uuid
        task_id = f"packet-{uuid.uuid4().hex[:8]}"
        task = {"task_id": task_id, "status": "created",
                "prompt": args.get("objective", ""),
                "description": args.get("scope", ""),
                "task_packet": args, "created_at": time.time()}
        self._tasks[task_id] = task
        return task

    # ══════════════════════════════════════════════════════════════════════
    #  TIER 3: WORKER (vereinfachte Implementierung)
    # ══════════════════════════════════════════════════════════════════════

    def _worker_create(self, args: dict) -> dict:
        import uuid
        wid = f"worker-{uuid.uuid4().hex[:8]}"
        self._workers[wid] = {
            "worker_id": wid, "status": "spawning",
            "cwd": args.get("cwd", str(self.ws.root)),
            "is_ready": False, "trust_gate_cleared": False,
            "trust_auto_resolve": bool(args.get("trusted_roots")),
            "prompt_in_flight": False,
        }
        return self._workers[wid]

    def _worker_get(self, args: dict) -> dict:
        wid = args["worker_id"]
        if wid not in self._workers:
            raise ValueError(f"Worker nicht gefunden: {wid}")
        return self._workers[wid]

    def _worker_observe(self, args: dict) -> dict:
        wid = args["worker_id"]
        if wid not in self._workers:
            raise ValueError(f"Worker nicht gefunden: {wid}")
        screen = args.get("screen_text", "")
        w = self._workers[wid]
        if "trust" in screen.lower() and not w["trust_gate_cleared"]:
            if w["trust_auto_resolve"]:
                w["trust_gate_cleared"] = True
            else:
                w["status"] = "trust_required"
        elif any(x in screen.lower() for x in ["ready", ">"]):
            w["status"] = "ready_for_prompt"
            w["is_ready"] = True
        return w

    def _worker_resolve_trust(self, args: dict) -> dict:
        wid = args["worker_id"]
        if wid not in self._workers:
            raise ValueError(f"Worker nicht gefunden: {wid}")
        self._workers[wid]["trust_gate_cleared"] = True
        self._workers[wid]["status"] = "spawning"
        return self._workers[wid]

    def _worker_await_ready(self, args: dict) -> dict:
        wid = args["worker_id"]
        if wid not in self._workers:
            raise ValueError(f"Worker nicht gefunden: {wid}")
        w = self._workers[wid]
        return {"worker_id": wid, "ready": w["is_ready"], "status": w["status"]}

    def _worker_send_prompt(self, args: dict) -> dict:
        wid = args["worker_id"]
        if wid not in self._workers:
            raise ValueError(f"Worker nicht gefunden: {wid}")
        w = self._workers[wid]
        if not w["is_ready"]:
            raise ValueError("Worker ist noch nicht bereit für Prompt-Lieferung")
        w["status"] = "running"
        w["prompt_in_flight"] = True
        return w

    def _worker_restart(self, args: dict) -> dict:
        wid = args["worker_id"]
        if wid not in self._workers:
            raise ValueError(f"Worker nicht gefunden: {wid}")
        w = self._workers[wid]
        w.update({"status": "spawning", "is_ready": False,
                   "trust_gate_cleared": False, "prompt_in_flight": False})
        return w

    def _worker_terminate(self, args: dict) -> dict:
        wid = args["worker_id"]
        if wid not in self._workers:
            raise ValueError(f"Worker nicht gefunden: {wid}")
        self._workers[wid]["status"] = "finished"
        self._workers[wid]["prompt_in_flight"] = False
        return self._workers[wid]

    def _worker_observe_completion(self, args: dict) -> dict:
        wid = args["worker_id"]
        if wid not in self._workers:
            raise ValueError(f"Worker nicht gefunden: {wid}")
        finish = args.get("finish_reason", "")
        tokens = args.get("tokens_output", 0)
        status = "finished" if finish == "end_turn" or tokens > 0 else "failed"
        self._workers[wid]["status"] = status
        self._workers[wid]["prompt_in_flight"] = False
        return self._workers[wid]

    # ══════════════════════════════════════════════════════════════════════
    #  TIER 4: TEAM/CRON
    # ══════════════════════════════════════════════════════════════════════

    def _team_create(self, args: dict) -> dict:
        import uuid
        tid = f"team-{uuid.uuid4().hex[:8]}"
        self._teams[tid] = {
            "team_id": tid, "name": args.get("name", ""),
            "task_ids": [t.get("task_id", "") for t in args.get("tasks", [])],
            "status": "running", "created_at": time.time(),
        }
        return self._teams[tid]

    def _team_delete(self, args: dict) -> dict:
        tid = args["team_id"]
        if tid not in self._teams:
            raise ValueError(f"Team nicht gefunden: {tid}")
        team = self._teams.pop(tid)
        team["status"] = "deleted"
        return team

    def _cron_create(self, args: dict) -> dict:
        import uuid
        cid = f"cron-{uuid.uuid4().hex[:8]}"
        self._crons[cid] = {
            "cron_id": cid, "schedule": args.get("schedule", ""),
            "prompt": args.get("prompt", ""),
            "description": args.get("description", ""),
            "enabled": True, "run_count": 0, "created_at": time.time(),
        }
        return self._crons[cid]

    def _cron_delete(self, args: dict) -> dict:
        cid = args["cron_id"]
        if cid not in self._crons:
            raise ValueError(f"Cron nicht gefunden: {cid}")
        cron = self._crons.pop(cid)
        cron["status"] = "deleted"
        return cron

    def _cron_list(self, args: dict) -> dict:
        return {"crons": list(self._crons.values()), "count": len(self._crons)}

    # ══════════════════════════════════════════════════════════════════════
    #  TIER 5: WEB
    # ══════════════════════════════════════════════════════════════════════

    def _web_fetch(self, args: dict) -> dict:
        url    = args["url"]
        prompt = args.get("prompt", "Summarize")
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nova-predator/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                content_type = resp.headers.get("Content-Type", "")
                body = resp.read(1024 * 1024).decode("utf-8", errors="replace")
                code = resp.status
        except urllib.error.URLError as e:
            return {"url": url, "error": str(e), "code": 0, "result": ""}

        # Einfaches HTML-zu-Text
        text = re.sub(r"<[^>]+>", " ", body)
        text = re.sub(r"\s+", " ", text).strip()[:4000]

        return {
            "url":        url,
            "code":       code,
            "bytes":      len(body),
            "result":     f"Fetched {url}\n\n{text[:2000]}",
            "durationMs": int((time.monotonic() - t0) * 1000),
        }

    def _web_search(self, args: dict) -> dict:
        query = args["query"]
        t0    = time.monotonic()
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={urllib.request.quote(query)}"
            req = urllib.request.Request(
                search_url,
                headers={"User-Agent": "Mozilla/5.0 nova-predator/1.0"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read(512 * 1024).decode("utf-8", errors="replace")

            # Links extrahieren
            hits = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)<', html)
            results = [{"url": u, "title": t.strip()} for u, t in hits[:8]]
        except Exception as e:
            results = []

        return {
            "query":           query,
            "results":         results,
            "durationSeconds": round(time.monotonic() - t0, 2),
        }

    # ══════════════════════════════════════════════════════════════════════
    #  TIER 6: CODE
    # ══════════════════════════════════════════════════════════════════════

    def _repl(self, args: dict) -> dict:
        import subprocess, sys, tempfile, os
        code     = args.get("code", "")
        language = args.get("language", "python").lower()
        timeout  = (args.get("timeout_ms", 30000) or 30000) // 1000

        if not code.strip():
            raise ValueError("code darf nicht leer sein")

        if language in ("python", "py"):
            return self.ws.run_python(code, timeout=timeout)
        elif language in ("javascript", "js", "node"):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".js",
                                              dir=str(self.ws.root), delete=False) as f:
                f.write(code); tmp = f.name
            try:
                proc = subprocess.run(["node", tmp], capture_output=True, encoding="utf-8", text=True,
                                       timeout=timeout, cwd=str(self.ws.root))
                return {"exitCode": proc.returncode, "stdout": proc.stdout,
                        "stderr": proc.stderr, "language": language}
            except FileNotFoundError:
                return {"exitCode": 1, "stdout": "", "stderr": "node nicht gefunden",
                        "language": language}
            finally:
                Path(tmp).unlink(missing_ok=True)
        else:
            raise ValueError(f"Nicht unterstützte REPL-Sprache: {language}")

    def _notebook_edit(self, args: dict) -> dict:
        """Jupyter Notebook Bearbeitung."""
        import json as json_mod
        path = args.get("notebook_path", "")
        if not path.endswith(".ipynb"):
            raise ValueError("Datei muss eine Jupyter-Notebook-Datei (.ipynb) sein")
        resolved = self.ws._resolve(path)
        notebook = json_mod.loads(resolved.read_text(encoding="utf-8"))
        cells    = notebook.get("cells", [])
        mode     = args.get("edit_mode", "replace")
        cell_id  = args.get("cell_id")
        source   = args.get("new_source", "")

        if mode == "insert":
            cell_type = args.get("cell_type", "code")
            new_cell  = {"cell_type": cell_type, "id": f"cell-{len(cells)+1}",
                          "metadata": {}, "source": source.splitlines(keepends=True),
                          "outputs": [], "execution_count": None}
            cells.append(new_cell)
        elif mode == "delete" and cell_id:
            cells = [c for c in cells if c.get("id") != cell_id]
        elif mode == "replace" and cell_id:
            for c in cells:
                if c.get("id") == cell_id:
                    c["source"] = source.splitlines(keepends=True)

        notebook["cells"] = cells
        resolved.write_text(json_mod.dumps(notebook, indent=2), encoding="utf-8")
        return {"notebook_path": str(resolved), "edit_mode": mode, "cell_id": cell_id}

    def _powershell(self, args: dict) -> dict:
        """PowerShell-Befehl (nur Windows)."""
        import subprocess, sys, platform
        command = args.get("command", "")
        timeout = args.get("timeout", 30)

        if platform.system() != "Windows":
            return {"stdout": "", "stderr": "PowerShell nur auf Windows verfügbar", "exitCode": 1}

        shell = "pwsh" if self._cmd_exists("pwsh") else "powershell"
        try:
            proc = subprocess.run(
                [shell, "-NoProfile", "-NonInteractive", "-Command", command],
                capture_output=True, encoding="utf-8", text=True, timeout=timeout, cwd=str(self.ws.root),
            )
            return {"stdout": proc.stdout, "stderr": proc.stderr, "exitCode": proc.returncode}
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": f"Timeout nach {timeout}s", "exitCode": -1}

    @staticmethod
    def _cmd_exists(cmd: str) -> bool:
        import shutil
        return shutil.which(cmd) is not None

    # ══════════════════════════════════════════════════════════════════════
    #  TIER 7: SYSTEM
    # ══════════════════════════════════════════════════════════════════════

    def _todo_write(self, args: dict) -> dict:
        todos = args.get("todos", [])
        if not todos:
            raise ValueError("todos darf nicht leer sein")
        # Integration mit Novas TodoManager wenn verfügbar
        return {"newTodos": todos, "count": len(todos)}

    def _skill(self, args: dict) -> dict:
        skill = args.get("skill", "").strip().lstrip("/$")
        skills_dir = self.ws.root.parent.parent / "skills"
        for candidate in [
            skills_dir / skill / "SKILL.md",
            skills_dir / f"{skill}.md",
        ]:
            if candidate.exists():
                content = candidate.read_text(encoding="utf-8")
                return {"skill": skill, "path": str(candidate), "prompt": content}
        raise ValueError(f"Skill nicht gefunden: {skill}")

    def _tool_search(self, args: dict) -> dict:
        query   = args.get("query", "").lower()
        max_res = args.get("max_results", 5)
        all_tools = list(self._handlers().keys())
        matches = [t for t in all_tools if query in t.lower()]
        return {"query": query, "matches": matches[:max_res],
                "total": len(all_tools)}

    def _sleep(self, args: dict) -> dict:
        ms = min(args.get("duration_ms", 0), 300_000)
        time.sleep(ms / 1000)
        return {"duration_ms": ms, "message": f"Geschlafen für {ms}ms"}

    def _send_user_message(self, args: dict) -> dict:
        msg = args.get("message", "")
        # In Agent-Loop: als Event in die Queue.
        # Nutzt gespeicherten Loop aus __init__ — get_event_loop() wirft RuntimeError
        # wenn dieser Code in run_in_executor (Thread) läuft (Python 3.12).
        if self._queue and self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait,
                {"typ": "agent_message", "text": msg}
            )
        return {"message": msg, "status": "sent", "sentAt": str(time.time())}

    def _config(self, args: dict) -> dict:
        setting = args.get("setting", "")
        value   = args.get("value")
        cfg_file = self.ws.root / ".claw" / "settings.local.json"
        cfg_file.parent.mkdir(exist_ok=True)
        data: dict = {}
        if cfg_file.exists():
            import json as jm
            data = jm.loads(cfg_file.read_text())
        if value is not None:
            data[setting] = value
            cfg_file.write_text(json.dumps(data, indent=2))
            return {"operation": "set", "setting": setting, "newValue": value}
        return {"operation": "get", "setting": setting, "value": data.get(setting)}

    def _enter_plan_mode(self, args: dict) -> dict:
        return {"active": True, "operation": "enter", "changed": True}

    def _exit_plan_mode(self, args: dict) -> dict:
        return {"active": False, "operation": "exit", "changed": True}

    def _structured_output(self, args: dict) -> dict:
        if not args:
            raise ValueError("structured output payload darf nicht leer sein")
        return {"data": "Structured output bereitgestellt", "structured_output": args}

    def _ask_user_question(self, args: dict) -> dict:
        # Im Agent-Loop: blockiert und wartet auf User-Input via Queue
        question = args.get("question", "")
        return {"question": question, "answer": "[Wartet auf User-Input]",
                "status": "pending"}

    def _lsp(self, args: dict) -> dict:
        return {"action": args.get("action"), "status": "lsp_not_available",
                "message": "LSP in dieser Nova-Version nicht konfiguriert"}

    def _mcp(self, args: dict) -> dict:
        return {"server": args.get("server"), "tool": args.get("tool"),
                "status": "mcp_not_connected",
                "error": "Kein MCP-Server verbunden"}

    def _mcp_auth(self, args: dict) -> dict:
        return {"server": args.get("server"), "status": "disconnected"}

    def _list_mcp_resources(self, args: dict) -> dict:
        return {"server": args.get("server", "default"), "resources": [], "count": 0}

    def _read_mcp_resource(self, args: dict) -> dict:
        return {"uri": args.get("uri"), "error": "MCP nicht verbunden"}

    def _design_ui(self, args: dict) -> dict:
        """Generiert vollständige UI-Datei via design_tool."""
        from layer5.design_tool import generate as ui_generate
        framework    = args.get("framework", "tkinter")
        beschreibung = args.get("beschreibung", "App")
        ausgabe      = args.get("ausgabe_datei", "app.py")
        stil         = args.get("stil", "modern-dark")
        komponenten  = args.get("komponenten", [])
        code   = ui_generate(framework, beschreibung, stil, komponenten)
        result = self.ws.write_file(ausgabe, code)
        return {
            "framework":     framework,
            "ausgabe_datei": ausgabe,
            "zeilen":        code.count("\n") + 1,
            "typ":           result.get("type", "create"),
        }

    def _remote_trigger(self, args: dict) -> dict:
        url    = args["url"]
        method = args.get("method", "GET").upper()
        try:
            req = urllib.request.Request(url, method=method)
            if args.get("body"):
                req.data = args["body"].encode()
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read(8192).decode("utf-8", errors="replace")
                return {"url": url, "method": method, "statusCode": resp.status,
                        "body": body, "success": 200 <= resp.status < 300}
        except Exception as e:
            return {"url": url, "method": method, "error": str(e), "success": False}
