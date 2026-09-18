"""Code-Sandbox-Skill: isolierte Python-Ausführung."""
from __future__ import annotations
import re
from core.skill_registry import SkillContext, SkillResult
from core.sandbox_guard import SandboxGuard

_guard = SandboxGuard()

def on_message(ctx: SkillContext) -> SkillResult | None:
    text = ctx.user_input

    # Code-Block aus Nachricht extrahieren
    code_match = re.search(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        # Inline-Code ohne Backticks
        zeilen = [z for z in text.splitlines() if z.strip() and not z.strip().startswith('#')]
        code_zeilen = [z for z in zeilen if any(kw in z for kw in
            ['=', 'print(', 'import ', 'def ', 'for ', 'while ', 'if ', 'return '])]
        if not code_zeilen or len(code_zeilen) < 1:
            return None  # Kein Code erkannt
        code = "\n".join(code_zeilen)

    timeout = ctx.skill_config.get("timeout", 30)
    result  = _guard.ausfuehren(code, timeout=timeout)

    if result["ok"]:
        output = result["output"].strip() or "(keine Ausgabe)"
        return SkillResult(
            inhalt=f"Code ausgeführt:\n```python\n{code[:500]}\n```\n\nAusgabe:\n```\n{output[:1000]}\n```",
            typ="aktion",
            metadaten={"code": code, "output": output}
        )
    else:
        fehler = result["fehler"].strip()
        return SkillResult(
            inhalt=f"Code-Fehler:\n```\n{fehler[:800]}\n```",
            typ="info",
            metadaten={"code": code, "fehler": fehler}
        )
