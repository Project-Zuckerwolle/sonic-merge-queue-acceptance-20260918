"""Nova Predator v1 — Workspace & Permissions (ehemals Layer 5).

Konsolidiert workspace_fs.py und permission.py aus Layer 5.
Nur was Apex tatsächlich braucht — kein toter Claw-spezifischer Code.

WorkspaceFS:  Boundary-checked Dateisystem-Zugriff (kein Path-Traversal)
PermissionMode: Welches Tool darf was tun
"""
from __future__ import annotations

from core.logger import get

log = get("layer6.workspace")

from enum import Enum
from pathlib import Path


# ── Permission-System ─────────────────────────────────────────────────────────

class PermissionMode(Enum):
    READ_ONLY       = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    DANGER_FULL     = "danger-full-access"


# Apex-Tool → benötigte Permission
TOOL_PERMISSIONS: dict[str, PermissionMode] = {
    # Lesen
    "datei_lesen":  PermissionMode.READ_ONLY,
    "glob_search":  PermissionMode.READ_ONLY,
    "grep_search":  PermissionMode.READ_ONLY,
    "brain_suche":  PermissionMode.READ_ONLY,
    # Schreiben (nur Workspace)
    "datei_schreiben": PermissionMode.WORKSPACE_WRITE,
    "edit_file":       PermissionMode.WORKSPACE_WRITE,
    "design_ui":       PermissionMode.WORKSPACE_WRITE,
    "notiz":           PermissionMode.WORKSPACE_WRITE,
    # Gefährlich (Shell)
    "bash":            PermissionMode.DANGER_FULL,
    "powershell":      PermissionMode.DANGER_FULL,
    "code_ausfuehren": PermissionMode.DANGER_FULL,
    # Netzwerk
    "websearch":       PermissionMode.READ_ONLY,
    "webfetch":        PermissionMode.READ_ONLY,
    # Skills/Briefing
    "skill_ausfuehren": PermissionMode.READ_ONLY,
    "briefing":         PermissionMode.READ_ONLY,
}


# ── Workspace Dateisystem ─────────────────────────────────────────────────────

MAX_READ_BYTES  = 10 * 1024 * 1024   # 10 MB
MAX_WRITE_BYTES = 10 * 1024 * 1024   # 10 MB


class WorkspaceBoundaryError(Exception):
    """Pfad verlässt die Workspace-Grenze."""


class WorkspaceFS:
    """Dateisystem-Zugriff streng innerhalb eines Workspace-Verzeichnisses.

    Verhindert Path-Traversal (../../evil.txt).
    Alle Pfade werden relativ zur root aufgelöst und geprüft.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str, must_exist: bool = True) -> Path:
        """Löst Pfad auf und prüft Workspace-Grenze."""
        p = Path(path)
        resolved = p.resolve() if p.is_absolute() else (self.root / path).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise WorkspaceBoundaryError(
                f"Pfad '{path}' liegt außerhalb der Workspace '{self.root}'"
            )
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {resolved}")
        return resolved

    def read(self, path: str) -> str:
        """Liest Datei, gibt Inhalt als String zurück."""
        resolved = self._resolve(path)
        size = resolved.stat().st_size
        if size > MAX_READ_BYTES:
            raise ValueError(f"Datei zu groß: {size // 1024}KB > {MAX_READ_BYTES // 1024}KB")
        return resolved.read_text(encoding="utf-8", errors="replace")

    def write(self, path: str, content: str) -> Path:
        """Schreibt Datei (überschreibt oder erstellt)."""
        if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
            raise ValueError("Inhalt zu groß (> 10MB)")
        resolved = self._resolve(path, must_exist=False)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return resolved

    def edit(self, path: str, old: str, new: str) -> int:
        """Ersetzt old durch new in Datei. Gibt Anzahl Ersetzungen zurück."""
        content = self.read(path)
        if old not in content:
            raise ValueError(f"String nicht gefunden in '{path}'")
        updated = content.replace(old, new, 1)
        self.write(path, updated)
        return 1

    def exists(self, path: str) -> bool:
        try:
            self._resolve(path)
            return True
        except (WorkspaceBoundaryError, FileNotFoundError):
            return False

    def list_files(self, pattern: str = "*") -> list[Path]:
        """Listet Dateien im Workspace (rekursiv, mit Glob-Filter)."""
        import fnmatch
        treffer = []
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and fnmatch.fnmatch(p.name, pattern):
                treffer.append(p.relative_to(self.root))
        return treffer[:100]
