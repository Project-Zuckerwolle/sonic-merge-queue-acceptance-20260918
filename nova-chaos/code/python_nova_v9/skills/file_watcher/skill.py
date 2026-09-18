"""FileWatcher-Skill: Workspace-Änderungen überwachen."""
from __future__ import annotations
import time
from pathlib import Path
from core.skill_registry import SkillContext, SkillResult

# Letzte bekannte Datei-Snapshots {pfad: mtime}
_snapshot: dict[str, float] = {}
_startup_zeit: float = 0.0


def on_startup(ctx: SkillContext) -> SkillResult | None:
    global _startup_zeit
    _startup_zeit = time.time()
    ws = Path(ctx.skill_config.get("workspace_pfad", "./workspace"))
    if ws.exists():
        for f in ws.rglob("*"):
            if f.is_file():
                _snapshot[str(f)] = f.stat().st_mtime
    return None  # Kein UI-Output beim Start


def on_message(ctx: SkillContext) -> SkillResult | None:
    ws  = Path(ctx.skill_config.get("workspace_pfad", "./workspace"))
    max_n = ctx.skill_config.get("max_anzeige", 10)

    if not ws.exists():
        return SkillResult(inhalt="Workspace-Verzeichnis existiert nicht.", typ="info")

    # Aktuelle Dateien scannen
    aktuell: dict[str, float] = {}
    for f in ws.rglob("*"):
        if f.is_file():
            aktuell[str(f)] = f.stat().st_mtime

    neue     = [p for p in aktuell if p not in _snapshot]
    geaendert= [p for p in aktuell if p in _snapshot and aktuell[p] > _snapshot[p]]
    geloescht= [p for p in _snapshot if p not in aktuell]

    # Snapshot aktualisieren
    _snapshot.update(aktuell)
    for p in geloescht:
        _snapshot.pop(p, None)

    if not neue and not geaendert and not geloescht:
        # Letzte geänderte Dateien anzeigen
        nach_zeit = sorted(aktuell.items(), key=lambda x: x[1], reverse=True)[:max_n]
        zeilen = [f"  {Path(p).name} ({_zeitstempel(m)})" for p, m in nach_zeit]
        return SkillResult(
            inhalt=f"Workspace ({len(aktuell)} Dateien). Zuletzt geändert:\n" + "\n".join(zeilen),
            typ="info"
        )

    teile = []
    if neue:
        teile.append(f"**Neu ({len(neue)}):**\n" + "\n".join(f"  + {Path(p).name}" for p in neue[:max_n]))
    if geaendert:
        teile.append(f"**Geändert ({len(geaendert)}):**\n" + "\n".join(f"  ~ {Path(p).name}" for p in geaendert[:max_n]))
    if geloescht:
        teile.append(f"**Gelöscht ({len(geloescht)}):**\n" + "\n".join(f"  - {Path(p).name}" for p in geloescht[:5]))

    return SkillResult(
        inhalt="Workspace-Änderungen:\n\n" + "\n\n".join(teile),
        typ="info",
        metadaten={"neu": len(neue), "geaendert": len(geaendert), "geloescht": len(geloescht)}
    )


def _zeitstempel(mtime: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(mtime).strftime("%d.%m %H:%M")
