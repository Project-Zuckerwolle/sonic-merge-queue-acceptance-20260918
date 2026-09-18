"""Nova Predator v1 Layer 5 — Permission System.

Spiegelt Claws PermissionMode direkt:
  ReadOnly          → nur lesen, keine Schreibzugriffe
  WorkspaceWrite    → Schreiben NUR in workspace/projektname/
  DangerFullAccess  → bash, REPL, PowerShell — erfordert explizite Freigabe
"""
from __future__ import annotations
from enum import Enum

from core.logger import get

log = get("layer5.permission")


class PermissionMode(Enum):
    READ_ONLY         = "read-only"
    WORKSPACE_WRITE   = "workspace-write"
    DANGER_FULL       = "danger-full-access"


# Welche Permission braucht welches Tool (aus Claws mvp_tool_specs)
TOOL_PERMISSIONS: dict[str, PermissionMode] = {
    # Tier 1 - Dateisystem
    "bash":            PermissionMode.DANGER_FULL,
    "read_file":       PermissionMode.READ_ONLY,
    "write_file":      PermissionMode.WORKSPACE_WRITE,
    "edit_file":       PermissionMode.WORKSPACE_WRITE,
    "glob_search":     PermissionMode.READ_ONLY,
    "grep_search":     PermissionMode.READ_ONLY,
    # Tier 2 - Agenten
    "Agent":           PermissionMode.DANGER_FULL,
    "TaskCreate":      PermissionMode.DANGER_FULL,
    "TaskGet":         PermissionMode.READ_ONLY,
    "TaskList":        PermissionMode.READ_ONLY,
    "TaskStop":        PermissionMode.DANGER_FULL,
    "TaskUpdate":      PermissionMode.DANGER_FULL,
    "TaskOutput":      PermissionMode.READ_ONLY,
    "RunTaskPacket":   PermissionMode.DANGER_FULL,
    # Tier 3 - Worker
    "WorkerCreate":         PermissionMode.DANGER_FULL,
    "WorkerGet":            PermissionMode.READ_ONLY,
    "WorkerObserve":        PermissionMode.READ_ONLY,
    "WorkerResolveTrust":   PermissionMode.DANGER_FULL,
    "WorkerAwaitReady":     PermissionMode.READ_ONLY,
    "WorkerSendPrompt":     PermissionMode.DANGER_FULL,
    "WorkerRestart":        PermissionMode.DANGER_FULL,
    "WorkerTerminate":      PermissionMode.DANGER_FULL,
    "WorkerObserveCompletion": PermissionMode.DANGER_FULL,
    # Tier 4 - Team/Cron
    "TeamCreate":      PermissionMode.DANGER_FULL,
    "TeamDelete":      PermissionMode.DANGER_FULL,
    "CronCreate":      PermissionMode.DANGER_FULL,
    "CronDelete":      PermissionMode.DANGER_FULL,
    "CronList":        PermissionMode.READ_ONLY,
    # Tier 5 - Web
    "WebFetch":        PermissionMode.READ_ONLY,
    "WebSearch":       PermissionMode.READ_ONLY,
    # Tier 6 - Code
    "REPL":            PermissionMode.DANGER_FULL,
    "NotebookEdit":    PermissionMode.WORKSPACE_WRITE,
    "PowerShell":      PermissionMode.DANGER_FULL,
    # Tier 7 - System
    "TodoWrite":       PermissionMode.WORKSPACE_WRITE,
    "Skill":           PermissionMode.READ_ONLY,
    "ToolSearch":      PermissionMode.READ_ONLY,
    "Sleep":           PermissionMode.READ_ONLY,
    "SendUserMessage": PermissionMode.READ_ONLY,
    "Config":          PermissionMode.WORKSPACE_WRITE,
    "EnterPlanMode":   PermissionMode.WORKSPACE_WRITE,
    "ExitPlanMode":    PermissionMode.WORKSPACE_WRITE,
    "StructuredOutput":PermissionMode.READ_ONLY,
    "AskUserQuestion": PermissionMode.READ_ONLY,
    "LSP":             PermissionMode.READ_ONLY,
    "MCP":             PermissionMode.DANGER_FULL,
    "McpAuth":         PermissionMode.DANGER_FULL,
    "ListMcpResources":PermissionMode.READ_ONLY,
    "ReadMcpResource": PermissionMode.READ_ONLY,
    "RemoteTrigger":   PermissionMode.DANGER_FULL,
    # Layer 5 Design-Tool
    "design_ui":       PermissionMode.WORKSPACE_WRITE,
}


class PermissionEnforcer:
    """Prüft ob ein Tool-Aufruf erlaubt ist.

    active_mode bestimmt die maximale erlaubte Permission:
      READ_ONLY       → nur ReadOnly-Tools
      WORKSPACE_WRITE → ReadOnly + WorkspaceWrite (default für Agent)
      DANGER_FULL     → alles erlaubt (muss explizit aktiviert werden)
    """

    def __init__(self, active_mode: PermissionMode = PermissionMode.WORKSPACE_WRITE,
                 allowed_tools: set[str] | None = None) -> None:
        self.active_mode   = active_mode
        self.allowed_tools = allowed_tools  # None = alle erlaubt

    def check(self, tool_name: str) -> tuple[bool, str]:
        """Gibt (erlaubt, grund) zurück."""
        # Allowlist prüfen
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            return False, f"Tool '{tool_name}' nicht in der Allowlist"

        required = TOOL_PERMISSIONS.get(tool_name, PermissionMode.DANGER_FULL)

        # Permission-Hierarchie
        order = [PermissionMode.READ_ONLY, PermissionMode.WORKSPACE_WRITE, PermissionMode.DANGER_FULL]
        active_idx  = order.index(self.active_mode)
        required_idx = order.index(required)

        if required_idx > active_idx:
            return False, (
                f"Tool '{tool_name}' benötigt {required.value}, "
                f"aktiver Modus ist {self.active_mode.value}"
            )
        return True, "ok"

    def enforce(self, tool_name: str) -> None:
        """Wirft ValueError wenn nicht erlaubt."""
        ok, grund = self.check(tool_name)
        if not ok:
            raise PermissionError(grund)
