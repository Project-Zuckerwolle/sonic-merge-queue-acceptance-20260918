"""Nova v8 – SandboxGuard.

AST-basierte Sicherheitsanalyse vor Code-Ausführung.
PYTHON 3.14 FIX: ast.Num/ast.Str/ast.Bytes entfernt → ast.Constant nutzen.
"""
from __future__ import annotations
import ast
import io
import re
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr

from core.logger import get

log = get("sandbox")

# Verbotene Builtins / Module
VERBOTEN_NAMEN = frozenset({
    "eval", "exec", "compile", "__import__", "open",
    "os", "sys", "subprocess", "shutil", "socket",
    "importlib", "ctypes", "pickle", "marshal",
    "__builtins__", "globals", "locals", "vars",
    "breakpoint", "memoryview",
})

VERBOTEN_IMPORTS = frozenset({
    "os", "sys", "subprocess", "shutil", "socket", "pathlib",
    "importlib", "ctypes", "pickle", "marshal", "multiprocessing",
    "threading", "concurrent", "asyncio", "signal", "atexit",
    "gc", "weakref", "__future__",
})

ERLAUBTE_IMPORTS = frozenset({
    "math", "json", "re", "datetime", "collections",
    "itertools", "functools", "string", "random",
    "statistics", "fractions", "decimal",
    "numpy", "np",  # numpy erlaubt
})


class SandboxFehler(Exception):
    pass


class SandboxGuard:
    """Prüft Python-Code via AST vor Ausführung."""

    def pruefen(self, code: str) -> None:
        """Wirft SandboxFehler wenn Code verboten ist."""
        try:
            baum = ast.parse(code)
        except SyntaxError as e:
            raise SandboxFehler(f"Syntaxfehler: {e}")

        self._pruefen_node(baum)

    def _pruefen_node(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            self._pruefen_einzel(child)

    def _pruefen_einzel(self, node: ast.AST) -> None:
        # Import-Prüfung
        if isinstance(node, ast.Import):
            for alias in node.names:
                modul = alias.name.split(".")[0]
                if modul in VERBOTEN_IMPORTS:
                    raise SandboxFehler(f"Import verboten: {alias.name}")
                if modul not in ERLAUBTE_IMPORTS and not modul.startswith("_"):
                    log.debug(f"Unbekannter Import: {alias.name} – erlaubt (kein Blocklist-Treffer)")

        elif isinstance(node, ast.ImportFrom):
            modul = (node.module or "").split(".")[0]
            if modul in VERBOTEN_IMPORTS:
                raise SandboxFehler(f"From-Import verboten: {node.module}")

        # Name-Prüfung (verbotene Builtins)
        elif isinstance(node, ast.Name):
            if node.id in VERBOTEN_NAMEN:
                raise SandboxFehler(f"Verbotener Name: {node.id}")

        # Call-Prüfung
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in VERBOTEN_NAMEN:
                    raise SandboxFehler(f"Verbotener Aufruf: {node.func.id}()")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in {"system", "popen", "exec_", "eval"}:
                    raise SandboxFehler(f"Verbotetes Attribut: .{node.func.attr}()")

        # Python 3.14: ast.Constant statt ast.Num/ast.Str
        # (ast.Num etc. wurden in 3.14 entfernt)
        elif isinstance(node, ast.Constant):
            # String-Injektion via __dunder__
            if isinstance(node.value, str) and node.value.startswith("__"):
                pass  # Warnung, kein Block – könnte legitim sein

        # Attribute-Zugriff auf __class__, __subclasses__ etc.
        elif isinstance(node, ast.Attribute):
            if node.attr in {"__class__", "__subclasses__", "__globals__", "__code__",
                             "__builtins__", "__import__"}:
                raise SandboxFehler(f"Verbotenes Attribut: .{node.attr}")


    def _preload_imports(self, code: str) -> tuple[dict, str]:
        """
        Findet erlaubte Imports, lädt sie vor, entfernt sie aus dem Code.
        Gibt (namespace_additions, bereinigter_code) zurück.
        """
        additions: dict = {}
        zeilen_entfernen: set[int] = set()
        zeilen = code.splitlines()

        import_re  = re.compile(r'^\s*import\s+([\w.]+)(?:\s+as\s+(\w+))?\s*$')
        from_re    = re.compile(r'^\s*from\s+([\w.]+)\s+import\s+(.+)\s*$')

        for i, zeile in enumerate(zeilen):
            m = import_re.match(zeile)
            if m:
                modul_name = m.group(1).split('.')[0]
                alias = m.group(2) or modul_name
                if modul_name in ERLAUBTE_IMPORTS:
                    try:
                        additions[alias] = __import__(modul_name)
                        # numpy-Sonderfall: np → numpy-Modul direkt
                        if modul_name == 'numpy' and alias == 'np':
                            import numpy as _np
                            additions['np'] = _np
                        zeilen_entfernen.add(i)
                    except ImportError:
                        zeilen_entfernen.add(i)  # trotzdem entfernen, wird im namespace fehlen
                continue

            m2 = from_re.match(zeile)
            if m2:
                modul_name = m2.group(1).split('.')[0]
                if modul_name in ERLAUBTE_IMPORTS:
                    try:
                        mod = __import__(modul_name)
                        additions[modul_name] = mod
                        # 'from x import a, b' → einzelne Attribute in namespace
                        importe = [s.strip().split(' as ') for s in m2.group(2).split(',')]
                        for imp in importe:
                            attr = imp[0].strip()
                            als = imp[1].strip() if len(imp) > 1 else attr
                            if hasattr(mod, attr):
                                additions[als] = getattr(mod, attr)
                        zeilen_entfernen.add(i)
                    except (ImportError, AttributeError):
                        zeilen_entfernen.add(i)

        bereinigte_zeilen = [z for i, z in enumerate(zeilen) if i not in zeilen_entfernen]
        return additions, '\n'.join(bereinigte_zeilen)

    def ausfuehren(self, code: str, timeout: int = 30) -> dict:
        """
        Führt Code sicher aus. Gibt {'output', 'fehler', 'ok'} zurück.
        """
        # Erst prüfen
        try:
            self.pruefen(code)
        except SandboxFehler as e:
            return {"output": "", "fehler": str(e), "ok": False}

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        # Sicherer __import__ - nur erlaubte Module
        def _sicherer_import(name, *args, **kwargs):
            basismodul = name.split('.')[0]
            if basismodul not in ERLAUBTE_IMPORTS:
                raise ImportError(f"Import nicht erlaubt: {name}")
            return __import__(name, *args, **kwargs)

        namespace = {
            "__builtins__": {
                "__import__": _sicherer_import,   # gesicherter Import
                "print": print, "len": len, "range": range, "enumerate": enumerate,
                "zip": zip, "map": map, "filter": filter, "sorted": sorted,
                "list": list, "dict": dict, "set": set, "tuple": tuple,
                "int": int, "float": float, "str": str, "bool": bool,
                "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
                "type": type, "isinstance": isinstance, "hasattr": hasattr,
                "getattr": getattr, "repr": repr, "format": format,
                "True": True, "False": False, "None": None,
                "Exception": Exception, "ValueError": ValueError,
                "TypeError": TypeError, "KeyError": KeyError,
            }
        }

        # Erlaubte Imports vorladen, aus Code entfernen
        import_additions, sauberer_code = self._preload_imports(code)
        namespace.update(import_additions)

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compile(sauberer_code, "<sandbox>", "exec"), namespace)
            ausgabe = stdout_buf.getvalue()
            fehler_text = stderr_buf.getvalue()
            return {"output": ausgabe, "fehler": fehler_text, "ok": True}
        except Exception as e:
            tb = traceback.format_exc(limit=5)
            return {"output": stdout_buf.getvalue(), "fehler": tb, "ok": False}
