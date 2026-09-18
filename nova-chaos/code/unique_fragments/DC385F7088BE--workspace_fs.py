"""Nova Predator v1 Layer 5 — Workspace Dateisystem.

Implementiert Claws file_ops.rs Logik in Python:
  - Boundary-Checks (kein Ausbruch aus workspace/)
  - read_file, write_file, edit_file, glob_search, grep_search
  - Alle Operationen nur innerhalb workspace/projektname/
"""
from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


MAX_READ_BYTES  = 10 * 1024 * 1024   # 10 MB
MAX_WRITE_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_GLOB_RESULTS = 100


class WorkspaceBoundaryError(Exception):
    """Pfad verlässt die Workspace-Grenze."""


class WorkspaceFS:
    """Dateisystem-Zugriff streng innerhalb von workspace/projekt/.

    Alle Pfade werden relativ zur root aufgelöst.
    Traversal (../../) wird erkannt und blockiert.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ── interne Helpers ────────────────────────────────────────────────────

    def _resolve(self, path: str, must_exist: bool = True) -> Path:
        """Löst Pfad auf und prüft Workspace-Grenze."""
        p = Path(path)
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (self.root / path).resolve()

        # Grenze prüfen
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise WorkspaceBoundaryError(
                f"Pfad '{path}' liegt außerhalb der Workspace '{self.root}'"
            )

        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {resolved}")

        return resolved

    def _is_binary(self, path: Path) -> bool:
        """Prüft ob Datei binär ist (enthält NUL-Bytes)."""
        try:
            with open(path, "rb") as f:
                chunk = f.read(8192)
            return b"\x00" in chunk
        except OSError:
            return False

    # ── Tool: read_file ────────────────────────────────────────────────────

    def read_file(self, path: str, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
        """Liest Datei, optional mit Zeilen-Fenster."""
        resolved = self._resolve(path)

        if resolved.stat().st_size > MAX_READ_BYTES:
            raise ValueError(f"Datei zu groß ({resolved.stat().st_size} Bytes, max {MAX_READ_BYTES})")

        if self._is_binary(resolved):
            raise ValueError("Datei ist binär und kann nicht gelesen werden")

        content = resolved.read_text(encoding="utf-8", errors="replace")
        lines   = content.splitlines()

        start = min(offset, len(lines))
        end   = len(lines) if limit is None else min(start + limit, len(lines))
        selected = "\n".join(lines[start:end])

        return {
            "type": "text",
            "file": {
                "filePath":   str(resolved),
                "content":    selected,
                "numLines":   end - start,
                "startLine":  start + 1,
                "totalLines": len(lines),
            },
        }

    # ── Tool: write_file ───────────────────────────────────────────────────

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        """Schreibt Datei (erstellt Verzeichnisse automatisch)."""
        if len(content.encode()) > MAX_WRITE_BYTES:
            raise ValueError(f"Inhalt zu groß (max {MAX_WRITE_BYTES} Bytes)")

        resolved = self._resolve(path, must_exist=False)
        original = resolved.read_text(encoding="utf-8", errors="replace") if resolved.exists() else None

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")

        return {
            "type":         "update" if original is not None else "create",
            "filePath":     str(resolved),
            "content":      content,
            "originalFile": original,
        }

    # ── Tool: edit_file ────────────────────────────────────────────────────

    def edit_file(self, path: str, old_string: str, new_string: str,
                  replace_all: bool = False) -> dict[str, Any]:
        """Ersetzt String in Datei."""
        resolved = self._resolve(path)
        original = resolved.read_text(encoding="utf-8", errors="replace")

        if old_string == new_string:
            raise ValueError("old_string und new_string müssen sich unterscheiden")
        if old_string not in original:
            raise ValueError(f"old_string nicht in Datei gefunden")

        updated = original.replace(old_string, new_string) if replace_all \
                  else original.replace(old_string, new_string, 1)
        resolved.write_text(updated, encoding="utf-8")

        return {
            "filePath":     str(resolved),
            "oldString":    old_string,
            "newString":    new_string,
            "originalFile": original,
            "replaceAll":   replace_all,
        }

    # ── Tool: append_file ─────────────────────────────────────────────────

    def append_file(self, path: str, content: str) -> dict[str, Any]:
        """Hängt Text an Datei an."""
        resolved = self._resolve(path, must_exist=False)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved, "a", encoding="utf-8") as f:
            f.write(content)
        return {"filePath": str(resolved), "appended": len(content)}

    # ── Tool: glob_search ─────────────────────────────────────────────────

    def glob_search(self, pattern: str, path: str | None = None) -> dict[str, Any]:
        """Findet Dateien per Glob-Pattern. Alle Ergebnisse bleiben in workspace."""
        t0 = time.monotonic()
        base = self._resolve(path, must_exist=True) if path else self.root

        matches: list[Path] = []

        # Primär: base.glob() mit dem vollen Pattern (unterstützt **/)
        try:
            raw = list(base.glob(pattern))
        except Exception:
            raw = []

        # Boundary-Check für JEDEN Treffer des primären Globs
        for p in raw:
            if not p.is_file():
                continue
            try:
                resolved = p.resolve()
                resolved.relative_to(self.root)   # wirft ValueError wenn außerhalb
                matches.append(resolved)
            except ValueError:
                pass  # außerhalb workspace → ignorieren

        # Fallback: rglob("*") + fnmatch nur wenn kein Treffer
        # (z.B. für einfache *.py patterns die base.glob nicht findet)
        if not matches:
            fname_pat = pattern.split("/")[-1]
            for p in base.rglob("*"):
                if not p.is_file():
                    continue
                if not fnmatch.fnmatch(p.name, fname_pat):
                    continue
                try:
                    resolved = p.resolve()
                    resolved.relative_to(self.root)
                    matches.append(resolved)
                except ValueError:
                    pass  # außerhalb workspace → ignorieren

        matches = sorted(set(matches), key=lambda p: p.stat().st_mtime, reverse=True)
        truncated = len(matches) > MAX_GLOB_RESULTS
        matches = matches[:MAX_GLOB_RESULTS]

        return {
            "durationMs": int((time.monotonic() - t0) * 1000),
            "numFiles":   len(matches),
            "filenames":  [str(p) for p in matches],
            "truncated":  truncated,
        }

    # ── Tool: grep_search ─────────────────────────────────────────────────

    def grep_search(self, pattern: str, path: str | None = None,
                    glob: str | None = None, case_insensitive: bool = False,
                    output_mode: str = "files_with_matches",
                    context: int = 0, head_limit: int | None = None) -> dict[str, Any]:
        """Durchsucht Dateien per Regex."""
        base = self._resolve(path, must_exist=True) if path else self.root

        try:
            flags = re.IGNORECASE if case_insensitive else 0
            regex = re.compile(pattern, flags)
        except re.error as e:
            raise ValueError(f"Ungültiges Regex: {e}")

        filenames: list[str] = []
        content_lines: list[str] = []
        total_matches = 0

        for file_path in sorted(base.rglob("*")):
            if not file_path.is_file():
                continue
            if glob and not fnmatch.fnmatch(file_path.name, glob):
                continue
            try:
                file_path.relative_to(self.root)
            except ValueError:
                continue
            if self._is_binary(file_path):
                continue

            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            lines = text.splitlines()
            matched_indices = [i for i, l in enumerate(lines) if regex.search(l)]

            if not matched_indices:
                continue

            filenames.append(str(file_path))
            total_matches += len(matched_indices)

            if output_mode == "content":
                for idx in matched_indices:
                    start = max(0, idx - context)
                    end   = min(len(lines), idx + context + 1)
                    for i in range(start, end):
                        content_lines.append(f"{file_path}:{i+1}:{lines[i]}")

        if head_limit:
            filenames = filenames[:head_limit]
            content_lines = content_lines[:head_limit]

        result: dict[str, Any] = {
            "numFiles":  len(filenames),
            "filenames": filenames,
        }
        if output_mode == "content":
            result["content"] = "\n".join(content_lines)
            result["numLines"] = len(content_lines)
        elif output_mode == "count":
            result["numMatches"] = total_matches

        return result

    # ── Tool: list_files ──────────────────────────────────────────────────

    def list_files(self, path: str | None = None, max_depth: int = 3) -> dict[str, Any]:
        """Gibt Dateibaum zurück (ohne Inhalte)."""
        base = self._resolve(path, must_exist=True) if path else self.root
        tree: list[dict[str, Any]] = []

        for p in sorted(base.rglob("*")):
            try:
                rel = p.relative_to(self.root)
            except ValueError:
                continue
            depth = len(rel.parts)
            if depth > max_depth:
                continue
            tree.append({
                "path":  str(rel),
                "isDir": p.is_dir(),
                "size":  p.stat().st_size if p.is_file() else 0,
            })

        return {"root": str(self.root), "files": tree, "count": len(tree)}

    # ── Tool: run_python ──────────────────────────────────────────────────

    def run_python(self, code: str, timeout: int = 30) -> dict[str, Any]:
        """Führt Python-Code in Sandbox aus (SandboxGuard + subprocess)."""
        # SandboxGuard-Check
        from core.sandbox_guard import SandboxGuard
        guard = SandboxGuard()
        result = guard.pruefen(code)
        if not result.erlaubt:
            return {
                "exitCode": 1,
                "stdout":   "",
                "stderr":   f"SandboxGuard: {result.grund}",
                "blocked":  True,
            }

        import sys
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                         dir=self.root, delete=False,
                                         encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name

        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, encoding="utf-8", text=True,
                timeout=timeout, cwd=str(self.root),
            )
            return {
                "exitCode":  proc.returncode,
                "stdout":    proc.stdout[:8192],
                "stderr":    proc.stderr[:4096],
                "blocked":   False,
            }
        except subprocess.TimeoutExpired:
            return {
                "exitCode": -1,
                "stdout":   "",
                "stderr":   f"Timeout nach {timeout}s",
                "blocked":  False,
            }
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ── Tool: run_tests ───────────────────────────────────────────────────

    def run_tests(self, test_path: str = ".", timeout: int = 120) -> dict[str, Any]:
        """Führt pytest im Workspace aus."""
        import sys
        resolved = self._resolve(test_path, must_exist=False)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", str(resolved), "-v", "--tb=short"],
                capture_output=True, text=True,
                timeout=timeout, cwd=str(self.root),
            )
            return {
                "exitCode": proc.returncode,
                "stdout":   proc.stdout[:16384],
                "stderr":   proc.stderr[:4096],
                "passed":   "passed" in proc.stdout,
            }
        except subprocess.TimeoutExpired:
            return {"exitCode": -1, "stdout": "", "stderr": f"Timeout nach {timeout}s", "passed": False}

    # ── Tool: install_package ─────────────────────────────────────────────

    PACKAGE_WHITELIST = {
        "requests", "flask", "fastapi", "uvicorn", "pydantic", "sqlalchemy",
        "pandas", "numpy", "matplotlib", "seaborn", "pillow", "pytest",
        "httpx", "aiohttp", "jinja2", "click", "typer", "rich", "tqdm",
        "python-dotenv", "pyyaml", "toml", "tomli", "cryptography",
        "paramiko", "boto3", "google-cloud-storage", "openai", "anthropic",
    }

    def install_package(self, package_name: str) -> dict[str, Any]:
        """Installiert Python-Paket (nur Whitelist)."""
        import sys
        base_name = package_name.split("==")[0].split(">=")[0].strip().lower()
        if base_name not in self.PACKAGE_WHITELIST:
            return {
                "success": False,
                "error":   f"Paket '{base_name}' nicht in der Whitelist",
                "whitelist": sorted(self.PACKAGE_WHITELIST),
            }

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True, encoding="utf-8", text=True, timeout=120,
            )
            return {
                "success":  proc.returncode == 0,
                "package":  package_name,
                "stdout":   proc.stdout[-2048:],
                "stderr":   proc.stderr[-1024:],
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout beim Installieren"}
