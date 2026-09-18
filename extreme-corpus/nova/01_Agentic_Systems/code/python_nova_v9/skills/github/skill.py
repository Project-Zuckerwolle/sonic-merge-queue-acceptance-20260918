"""GitHub-Skill: Git-Operationen."""
from __future__ import annotations
import subprocess
from pathlib import Path
from core.skill_registry import SkillContext, SkillResult

def on_message(ctx: SkillContext) -> SkillResult | None:
    ws = Path(ctx.skill_config.get("workspace_pfad", "./workspace"))
    text = ctx.user_input.lower()

    # Aktuellsten Task-Ordner finden
    ordner = ws
    if ws.exists():
        task_dirs = sorted([d for d in ws.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
        if task_dirs:
            ordner = task_dirs[0]

    if not ordner.exists():
        return SkillResult(inhalt="Kein Workspace-Ordner gefunden.", typ="info")

    try:
        if "status" in text:
            r = subprocess.run(["git", "status"], cwd=ordner, capture_output=True, text=True, timeout=10)
            return SkillResult(inhalt=f"Git Status:\n```\n{r.stdout[:1000]}\n```", typ="info")

        if "commit" in text or "push" in text:
            # Commit
            subprocess.run(["git", "add", "."], cwd=ordner, capture_output=True, timeout=10)
            msg_match = __import__("re").search(r'"([^"]+)"', ctx.user_input)
            msg = msg_match.group(1) if msg_match else "Nova: Update"
            r = subprocess.run(["git", "commit", "-m", msg], cwd=ordner, capture_output=True, text=True, timeout=10)
            output = r.stdout.strip() or r.stderr.strip()
            if "push" in text:
                r2 = subprocess.run(["git", "push"], cwd=ordner, capture_output=True, text=True, timeout=30)
                output += f"\nPush: {r2.stdout.strip() or r2.stderr.strip()}"
            return SkillResult(inhalt=f"Git:\n```\n{output[:800]}\n```", typ="aktion")

        if "log" in text:
            r = subprocess.run(["git", "log", "--oneline", "-10"], cwd=ordner, capture_output=True, text=True, timeout=10)
            return SkillResult(inhalt=f"Git Log:\n```\n{r.stdout[:1000]}\n```", typ="info")

    except Exception as e:
        return SkillResult(inhalt=f"Git-Fehler: {e}", typ="info")

    return None
