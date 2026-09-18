"""Datei-Zugriff-Skill: Workspace-Dateien lesen/auflisten."""
from __future__ import annotations
import re
from pathlib import Path
from core.skill_registry import SkillContext, SkillResult

def on_message(ctx: SkillContext) -> SkillResult | None:
    ws_pfad  = Path(ctx.skill_config.get("workspace_pfad", "./workspace"))
    max_kb   = ctx.skill_config.get("max_dateigroesse_kb", 100)
    text     = ctx.user_input.lower()

    # Dateinamen aus Anfrage extrahieren
    datei_match = re.search(r'[\w\-./]+\.\w{1,6}', ctx.user_input)

    if datei_match:
        # Datei lesen
        datei_name = datei_match.group()
        # Sicherheit: nur im Workspace
        pfad = (ws_pfad / datei_name).resolve()
        ws_resolved = ws_pfad.resolve()
        if not str(pfad).startswith(str(ws_resolved)):
            return SkillResult(inhalt="Sicherheit: Zugriff nur auf Workspace-Dateien erlaubt.", typ="info")
        if not pfad.exists():
            # Auch im aktuellen Verzeichnis suchen
            pfad2 = Path(datei_name).resolve()
            if pfad2.exists() and pfad2.stat().st_size < max_kb * 1024:
                pfad = pfad2
            else:
                return SkillResult(inhalt=f"Datei nicht gefunden: {datei_name}", typ="info")
        if pfad.stat().st_size > max_kb * 1024:
            return SkillResult(
                inhalt=f"Datei zu groß ({pfad.stat().st_size//1024}KB > {max_kb}KB): {datei_name}",
                typ="info"
            )
        try:
            inhalt = pfad.read_text(encoding="utf-8", errors="replace")
            return SkillResult(
                inhalt=f"Inhalt von {datei_name}:\n\n```\n{inhalt[:3000]}\n```",
                typ="datei", metadaten={"pfad": str(pfad)}
            )
        except Exception as e:
            return SkillResult(inhalt=f"Lesefehler: {e}", typ="info")
    else:
        # Verzeichnis auflisten
        if not ws_pfad.exists():
            return SkillResult(inhalt="Workspace-Verzeichnis leer oder nicht vorhanden.", typ="info")
        dateien = []
        for f in sorted(ws_pfad.rglob("*"))[:50]:
            if f.is_file():
                groesse = f.stat().st_size
                rel     = f.relative_to(ws_pfad)
                dateien.append(f"  {rel} ({groesse//1024}KB)" if groesse > 1024 else f"  {rel}")
        if not dateien:
            return SkillResult(inhalt="Workspace ist leer.", typ="info")
        return SkillResult(
            inhalt=f"Dateien im Workspace ({len(dateien)}):\n" + "\n".join(dateien),
            typ="info"
        )
