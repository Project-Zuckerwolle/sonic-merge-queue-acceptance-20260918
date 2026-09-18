"""Python-REPL-Skill: persistenter State zwischen Aufrufen.

Im Gegensatz zur code_sandbox bleibt hier der Namespace erhalten —
Variablen, Funktionen und Imports leben über mehrere Nachrichten.
"""
from __future__ import annotations
import ast
import io
import re
import traceback
from contextlib import redirect_stdout, redirect_stderr
from core.skill_registry import SkillContext, SkillResult
from core.sandbox_guard import ERLAUBTE_IMPORTS, VERBOTEN_NAMEN

# Persistenter Namespace (lebt solange der Server läuft)
_namespace: dict = {}
_history:   list[str] = []

# Sicherer __import__ für REPL
def _repl_import(name, *args, **kwargs):
    if name.split('.')[0] not in ERLAUBTE_IMPORTS:
        raise ImportError(f"Import nicht erlaubt: {name}")
    return __import__(name, *args, **kwargs)

def _init_namespace() -> dict:
    """Initialisiert oder gibt bestehenden Namespace zurück."""
    if "__builtins__" not in _namespace:
        _namespace["__builtins__"] = {
            "__import__": _repl_import,
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
    return _namespace


def on_message(ctx: SkillContext) -> SkillResult | None:
    text = ctx.user_input

    # REPL-Trigger erkennen
    repl_match = re.search(r'(?:repl:|python repl:)\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    code_block  = re.search(r'```(?:python)?\n(.*?)```', text, re.DOTALL)

    if repl_match:
        code = repl_match.group(1).strip()
    elif code_block:
        code = code_block.group(1).strip()
    else:
        # Inline-Code erkennen (Zeilen die wie Code aussehen)
        zeilen = text.splitlines()
        code_zeilen = [
            z for z in zeilen
            if re.match(r'\s*(?:\w+\s*=|print\(|def |class |for |if |import |from )', z)
        ]
        if not code_zeilen:
            return None
        code = "\n".join(code_zeilen)

    if not code:
        return None

    # AST-Sicherheitscheck
    try:
        baum = ast.parse(code)
    except SyntaxError as e:
        return SkillResult(inhalt=f"Syntaxfehler: {e}", typ="info")

    for node in ast.walk(baum):
        if isinstance(node, ast.Name) and node.id in VERBOTEN_NAMEN:
            return SkillResult(inhalt=f"Verbotener Name: {node.id}", typ="info")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modul = ""
            if isinstance(node, ast.Import):
                modul = node.names[0].name.split(".")[0]
            elif node.module:
                modul = node.module.split(".")[0]
            if modul and modul not in ERLAUBTE_IMPORTS:
                return SkillResult(inhalt=f"Import nicht erlaubt: {modul}", typ="info")

    # Code ausführen mit persistentem Namespace
    ns = _init_namespace()
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(compile(code, "<repl>", "exec"), ns)

        output  = stdout_buf.getvalue().strip()
        fehler  = stderr_buf.getvalue().strip()

        # History
        _history.append(code)
        if len(_history) > ctx.skill_config.get("max_history", 50):
            _history.pop(0)

        # Welche Variablen sind jetzt im Namespace?
        vars_neu = {
            k: repr(v)[:50]
            for k, v in ns.items()
            if not k.startswith("_") and k != "__builtins__"
        }

        antwort = f"```python\n{code}\n```\n"
        if output:
            antwort += f"\n**Ausgabe:**\n```\n{output[:1000]}\n```"
        if fehler:
            antwort += f"\n**Stderr:**\n```\n{fehler[:300]}\n```"
        if vars_neu:
            vars_str = ", ".join(f"`{k}={v}`" for k, v in list(vars_neu.items())[:8])
            antwort += f"\n\n**State:** {vars_str}"
        antwort += f"\n*(REPL: {len(_history)} Befehle in History)*"

        return SkillResult(
            inhalt=antwort, typ="aktion",
            metadaten={"code": code, "output": output, "namespace_size": len(vars_neu)}
        )

    except Exception as e:
        tb = traceback.format_exc(limit=4)
        return SkillResult(
            inhalt=f"REPL-Fehler:\n```\n{tb[:600]}\n```",
            typ="info"
        )


def repl_reset() -> None:
    """Setzt den REPL-State zurück (für Tests / manuellen Reset)."""
    _namespace.clear()
    _history.clear()
