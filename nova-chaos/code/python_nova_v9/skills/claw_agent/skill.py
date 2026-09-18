"""Claw-Agent-Skill: autonomer Coding-Agent.

Korrekte Nutzung von codetwentyfive/claw-code-local:
  - Binary: claw (aus cargo build -p rusty-claude-cli --release)
  - CLI:    claw --model <modell> "<aufgabe>"
  - EnvVars: ANTHROPIC_API_KEY (Dummy-Wert reicht für Ollama)
             ANTHROPIC_BASE_URL (Ollama OpenAI-compat Endpunkt)
"""
from __future__ import annotations
import os
import platform
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from core.skill_registry import SkillContext, SkillResult
from core.event_bus import bus, EventTyp

_aktive_tasks: dict[str, dict] = {}


def on_message(ctx: SkillContext) -> SkillResult | None:
    binary = _finde_claw(ctx.skill_config.get("binary_pfad", ""))

    if not binary:
        return SkillResult(
            inhalt=(
                "Claw-Binary nicht gefunden.\n\n"
                "**Installation:**\n"
                "```bash\n"
                "git clone https://github.com/codetwentyfive/claw-code-local\n"
                "cd claw-code-local/rust\n"
                "cargo build -p rusty-claude-cli --release\n"
                "```\n"
                "Dann in config.yaml:\n"
                "```yaml\n"
                "skills:\n"
                "  ClawAgent:\n"
                "    binary_pfad: ./claw-code-local/rust/target/release/claw\n"
                "```"
            ),
            typ="info",
        )

    # Ollama-Host aus Config
    ollama_host = ctx.config.get("ollama", {}).get("host", "http://localhost:11434")
    modell      = ctx.config.get("modelle", {}).get("code") or ctx.config.get("ollama", {}).get("modell", "gemma4:e4b")

    ws      = Path(ctx.skill_config.get("workspace_pfad", "./workspace"))
    aufgabe = ctx.user_input.strip()
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug    = "".join(c if c.isalnum() else "_" for c in aufgabe[:25])
    ordner  = ws / f"task_{ts}_{slug}"
    ordner.mkdir(parents=True, exist_ok=True)

    task_id = ts
    _aktive_tasks[task_id] = {
        "ordner":  str(ordner),
        "aufgabe": aufgabe,
        "output":  [],
        "fertig":  False,
        "fehler":  "",
    }

    # Umgebungsvariablen für Claw:
    # - ANTHROPIC_API_KEY: muss gesetzt + nicht-leer sein (Wert egal für Ollama)
    # - ANTHROPIC_BASE_URL: Ollama OpenAI-compat Endpunkt (/v1 Suffix nötig)
    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"]  = env.get("ANTHROPIC_API_KEY", "ollama-local")
    env["ANTHROPIC_BASE_URL"] = f"{ollama_host.rstrip('/')}/v1"

    # Korrekte CLI-Syntax: claw --model <modell> "<aufgabe>"
    cmd = [binary, "--model", modell, aufgabe]

    def _run():
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ordner),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                encoding="utf-8",
                errors="replace",
            )
            for zeile in proc.stdout:
                zeile = zeile.rstrip()
                if zeile:
                    _aktive_tasks[task_id]["output"].append(zeile)
                    # Live-Output an EventBus → WebSocket
                    bus.publish_threadsafe(
                        EventTyp.AGENT_OUTPUT,
                        {"task_id": task_id, "zeile": zeile},
                    )

            proc.wait(timeout=600)
            _aktive_tasks[task_id]["fertig"]     = True
            _aktive_tasks[task_id]["returncode"] = proc.returncode

            # Fertig-Event
            erfolg = proc.returncode == 0
            bus.publish_threadsafe(
                EventTyp.AGENT_FERTIG,
                {
                    "task_id":      task_id,
                    "ordner":       str(ordner),
                    "aufgabe":      aufgabe,
                    "erfolg":       erfolg,
                    "returncode":   proc.returncode,
                    "zusammenfassung": f"{'OK' if erfolg else 'Fehler'}: {aufgabe[:80]}",
                },
            )
        except subprocess.TimeoutExpired:
            _aktive_tasks[task_id]["fehler"] = "Timeout nach 10 Minuten"
            _aktive_tasks[task_id]["fertig"] = True
            bus.publish_threadsafe(EventTyp.AGENT_FERTIG,
                {"task_id": task_id, "erfolg": False, "zusammenfassung": "Timeout"})
        except Exception as e:
            _aktive_tasks[task_id]["fehler"] = str(e)
            _aktive_tasks[task_id]["fertig"] = True
            bus.publish_threadsafe(EventTyp.AGENT_FERTIG,
                {"task_id": task_id, "erfolg": False, "zusammenfassung": str(e)})

    threading.Thread(target=_run, daemon=True, name=f"claw_{task_id}").start()

    return SkillResult(
        inhalt=(
            f"Claw-Agent gestartet.\n"
            f"Modell: {modell}\n"
            f"Aufgabe: {aufgabe[:100]}\n"
            f"Arbeitsordner: {ordner.name}\n"
            f"Ollama: {ollama_host}"
        ),
        typ="stream",
        weiter_aktiv=True,
        metadaten={"task_id": task_id, "ordner": str(ordner), "aufgabe": aufgabe},
    )


def _finde_claw(config_pfad: str) -> str | None:
    """
    Sucht das Claw-Binary in mehreren Pfaden.
    Unterstützt Windows (where) und Unix (which).
    """
    on_windows = platform.system() == "Windows"

    # 1. Expliziter Pfad aus config.yaml
    if config_pfad:
        p = Path(config_pfad)
        if p.exists():
            return str(p)

    # 2. Bekannte Build-Pfade (codetwentyfive + ultraworkers)
    bekannte = [
        # codetwentyfive/claw-code-local (Rust crate: rusty-claude-cli → Binary: claw)
        "./claw-code-local/rust/target/release/claw",
        "./claw-code-local/rust/target/release/claw.exe",
        # ultraworkers/claw-code
        "./claw-code/rust/target/release/claw",
        "./claw-code/rust/target/release/claw.exe",
        # Im PATH
        "claw",
        "claw.exe",
    ]
    for kandidat in bekannte:
        p = Path(kandidat)
        if p.exists() and p.is_file():
            return str(p)

    # 3. PATH-Suche (plattformspezifisch)
    such_cmd = ["where", "claw"] if on_windows else ["which", "claw"]
    try:
        r = subprocess.run(such_cmd, capture_output=True, text=True, timeout=3)
        if r.returncode == 0:
            pfad = r.stdout.strip().splitlines()[0].strip()
            if pfad:
                return pfad
    except Exception:
        pass

    return None


# ─── Hilfsfunktionen für app.py ───────────────────────────────────────────────
def task_status(task_id: str) -> dict:
    return _aktive_tasks.get(task_id, {})

def alle_tasks() -> dict:
    return dict(_aktive_tasks)

def task_abbrechen(task_id: str) -> bool:
    """Versucht einen laufenden Task zu beenden."""
    task = _aktive_tasks.get(task_id)
    if not task or task.get("fertig"):
        return False
    task["fehler"] = "Manuell abgebrochen"
    task["fertig"] = True
    return True
