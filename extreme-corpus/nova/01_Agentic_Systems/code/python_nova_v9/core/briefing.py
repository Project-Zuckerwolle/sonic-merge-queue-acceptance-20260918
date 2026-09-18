"""Nova v8 – Briefing.

Generiert tägliche und wöchentliche Zusammenfassungen
aus Brain, Todos und Session-History.
"""
from __future__ import annotations
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from core.logger import get
from core.event_bus import bus, EventTyp

if TYPE_CHECKING:
    from core.ollama_client import OllamaClient
    from core.brain_manager import BrainManager
    from core.todo_manager  import TodoManager

log = get("briefing")

_DAILY_PROMPT = """Erstelle ein kurzes tägliches Briefing für den Nutzer (3-5 Sätze).
Basiere es auf den folgenden Informationen:

Offene Todos ({todo_anzahl}):
{todos}

Aktuelle Brain-Einträge (letzte 5):
{brain}

Heutiges Datum: {datum}

Schreibe das Briefing freundlich, direkt und auf Deutsch."""

_WEEKLY_PROMPT = """Erstelle eine wöchentliche Zusammenfassung (5-8 Sätze).
Was wurde diese Woche gelernt? Was steht noch aus?

Todos diese Woche:
{todos}

Brain-Einträge diese Woche ({anzahl} neu):
{brain}

Woche: {datum}"""


class Briefing:
    def __init__(self, ollama: "OllamaClient", brain: "BrainManager", todos: "TodoManager",
                 pfad: str = "./data/briefings") -> None:
        self._ollama = ollama
        self._brain  = brain
        self._todos  = todos
        self._pfad   = Path(pfad)
        self._pfad.mkdir(parents=True, exist_ok=True)

    # ─── Daily Briefing ───────────────────────────────────────────────────────
    async def generiere_daily(self) -> str:
        heute = datetime.now().strftime("%Y-%m-%d")
        datei = self._pfad / f"daily_{heute}.txt"
        if datei.exists():
            return datei.read_text(encoding="utf-8")

        todos_offen = self._todos.offen()[:5]
        todos_text  = "\n".join(f"- [{t['prioritaet']}] {t['text']}" for t in todos_offen) or "Keine offenen Todos."

        index = self._brain.get_index()
        letzte_entries = sorted(index, key=lambda e: e.get("erstellt",""), reverse=True)[:5]
        brain_text = "\n".join(f"- {e['titel']} ({e['typ']})" for e in letzte_entries) or "Noch keine Brain-Einträge."

        prompt = _DAILY_PROMPT.format(
            todo_anzahl=len(todos_offen),
            todos=todos_text,
            brain=brain_text,
            datum=datetime.now().strftime("%A, %d. %B %Y"),
        )
        text = self._ollama.generiere([{"role": "user", "content": prompt}], max_tokens=300)
        if not text:
            text = f"Guten Morgen! Heute ist {heute}. Du hast {len(todos_offen)} offene Todos."

        datei.write_text(text, encoding="utf-8")
        await bus.publish(EventTyp.BRIEFING_GENERIERT, {"typ": "daily", "datum": heute})
        log.info(f"Daily Briefing generiert: {heute}")
        return text

    # ─── Weekly Briefing ─────────────────────────────────────────────────────
    async def generiere_weekly(self) -> str:
        kw = datetime.now().isocalendar()[1]
        jahr = datetime.now().year
        datei = self._pfad / f"weekly_{jahr}_KW{kw:02d}.txt"
        if datei.exists():
            return datei.read_text(encoding="utf-8")

        woche_start = datetime.now() - timedelta(days=7)
        index = self._brain.get_index()
        neue_entries = [
            e for e in index
            if e.get("erstellt","") >= woche_start.isoformat()
        ]
        brain_text = "\n".join(f"- {e['titel']}" for e in neue_entries[:10]) or "Keine neuen Einträge."

        todos_alle = self._todos.alle()
        todos_woche = [
            t for t in todos_alle
            if (t.get("erstellt","") >= woche_start.isoformat() or
                t.get("erledigt_am","") and t["erledigt_am"] >= woche_start.isoformat())
        ]
        todos_text = "\n".join(
            f"- [{'✓' if t['erledigt'] else '○'}] {t['text']}" for t in todos_woche[:8]
        ) or "Keine Todo-Aktivität diese Woche."

        prompt = _WEEKLY_PROMPT.format(
            todos=todos_text, brain=brain_text,
            anzahl=len(neue_entries),
            datum=f"KW {kw}, {jahr}",
        )
        text = self._ollama.generiere([{"role": "user", "content": prompt}], max_tokens=400)
        if not text:
            text = f"Woche {kw}/{jahr}: {len(neue_entries)} neue Brain-Einträge."

        datei.write_text(text, encoding="utf-8")
        log.info(f"Weekly Briefing generiert: KW{kw}/{jahr}")
        return text

    def heute_schon_generiert(self) -> bool:
        heute = datetime.now().strftime("%Y-%m-%d")
        return (self._pfad / f"daily_{heute}.txt").exists()

    def liste_briefings(self) -> list[str]:
        return sorted([p.name for p in self._pfad.glob("*.txt")], reverse=True)
