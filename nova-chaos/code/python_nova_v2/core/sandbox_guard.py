"""Nova Predator v1 — SandboxGuard.
Sicherheits-Analyse für Python-Code vor Ausführung.
Python 3.14: ast.Constant (ast.Num/Str deprecated seit 3.8, entfernt in 3.12).
Kein Nova-Import außer logger.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass, field
from typing import Any

from core.logger import get

log = get("sandbox_guard")

_VERBOTENE_IMPORTE = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "ctypes", "importlib", "builtins", "exec", "eval",
    "open", "__import__",
})

_VERBOTENE_CALLS = frozenset({
    "exec", "eval", "compile", "__import__", "open",
    "input", "breakpoint",
})


@dataclass
class PrüfErgebnis:
    erlaubt: bool
    grund: str = ""
    warnungen: list[str] = field(default_factory=list)


class SandboxGuard:
    def pruefen(self, code: str) -> PrüfErgebnis:
        """Analysiert Code auf gefährliche Konstrukte."""
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as e:
            return PrüfErgebnis(erlaubt=False, grund=f"Syntaxfehler: {e}")

        besucher = _CodeBesucher()
        besucher.visit(tree)

        if besucher.verbote:
            return PrüfErgebnis(
                erlaubt=False,
                grund=f"Verbotene Konstrukte: {', '.join(besucher.verbote)}",
                warnungen=besucher.warnungen,
            )

        return PrüfErgebnis(
            erlaubt=True,
            warnungen=besucher.warnungen,
        )


class _CodeBesucher(ast.NodeVisitor):
    def __init__(self) -> None:
        self.verbote: list[str] = []
        self.warnungen: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            basis = alias.name.split(".")[0]
            if basis in _VERBOTENE_IMPORTE:
                self.verbote.append(f"import {basis}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            basis = node.module.split(".")[0]
            if basis in _VERBOTENE_IMPORTE:
                self.verbote.append(f"from {basis}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Direkte Calls: eval(), exec()
        if isinstance(node.func, ast.Name):
            if node.func.id in _VERBOTENE_CALLS:
                self.verbote.append(f"call:{node.func.id}()")
        # Attribut-Calls: os.system() etc.
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in ("system", "popen", "run", "check_output", "Popen"):
                self.warnungen.append(f"Verdächtig: .{node.func.attr}()")
        self.generic_visit(node)

    # Python 3.14: ast.Constant für alle Literale (ast.Num/Str entfernt in 3.12)
    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and "__import__" in node.value:
            self.warnungen.append("String enthält __import__")
        self.generic_visit(node)
