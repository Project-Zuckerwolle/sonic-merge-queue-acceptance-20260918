"""Nova Predator v1 Layer 6 — Routinen-System.

Lädt benannte Routinen aus routinen/*.yaml.
Registriert sie im bestehenden Scheduler (Layer 0).
Jede Routine kann Skills aufrufen, Apps starten oder den Apex-Agent starten.

YAML-Format (routinen/beispiel.yaml):
  name: morgen_routine
  trigger:
    typ: uhrzeit        # uhrzeit | intervall | manuell
    zeit: "07:30"
    wochentage: [Mo, Di, Mi, Do, Fr]   # optional
  aktionen:
    - typ: skill
      skill: wetter
      user_input: "Wetter heute Berlin"
    - typ: skill
      skill: news
      user_input: "Aktuelle Nachrichten"
    - typ: briefing      # Daily Briefing generieren
    - typ: tts           # Ergebnis vorlesen
      text: "Guten Morgen! Hier ist dein Briefing:"
    - typ: app_starten
      app: chrome
    - typ: apex          # Apex-Agent für komplexe Aufgaben
      aufgabe: "Recherchiere tech news der letzten 24h"
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, TYPE_CHECKING

import yaml

from core.logger import get

if TYPE_CHECKING:
    from core.scheduler import Scheduler
    from layer6.apex_orchestrator import ApexOrchestrator

log = get("apex.routinen")


@dataclass
class Routine:
    name:         str
    trigger_typ:  str            # uhrzeit | intervall | manuell
    trigger_zeit: str | None     # "HH:MM" für uhrzeit
    trigger_s:    float | None   # Sekunden für intervall
    wochentage:   list[str]      # [] = alle Tage
    aktionen:     list[dict]     # Aktionsliste
    aktiv:        bool = True
    letzter_run:  str | None = None


WOCHENTAG_MAP = {
    "mo": 0, "di": 1, "mi": 2, "do": 3, "fr": 4, "sa": 5, "so": 6,
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}


class RoutinenManager:
    """Verwaltet und führt Routinen aus."""

    def __init__(
        self,
        routinen_pfad: str | Path = "routinen",
        scheduler: "Scheduler | None" = None,
        orchestrator: "ApexOrchestrator | None" = None,
    ) -> None:
        self._pfad        = Path(routinen_pfad)
        self._scheduler   = scheduler
        self._orch        = orchestrator
        self._routinen:   dict[str, Routine] = {}

    async def alle_laden(self) -> int:
        """Lädt alle Routinen aus dem routinen/-Verzeichnis."""
        self._pfad.mkdir(parents=True, exist_ok=True)
        geladen = 0

        for yaml_pfad in sorted(self._pfad.glob("*.yaml")):
            routine = self._lade_yaml(yaml_pfad)
            if routine:
                self._routinen[routine.name] = routine
                await self._registriere_in_scheduler(routine)
                geladen += 1
                log.debug("Routine geladen: %s (%s)", routine.name, routine.trigger_typ)

        log.info("Routinen geladen: %d", geladen)
        return geladen

    def _lade_yaml(self, pfad: Path) -> Routine | None:
        """Parst eine Routine-YAML-Datei."""
        try:
            with open(pfad, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not data or "name" not in data or "aktionen" not in data:
                log.warning("Ungültige Routine-YAML: %s", pfad.name)
                return None

            trigger = data.get("trigger", {})
            trigger_typ  = trigger.get("typ", "manuell")
            trigger_zeit = trigger.get("zeit")
            trigger_s    = trigger.get("intervall_s")
            wochentage   = [
                d.lower() for d in trigger.get("wochentage", [])
            ]

            return Routine(
                name         = data["name"],
                trigger_typ  = trigger_typ,
                trigger_zeit = trigger_zeit,
                trigger_s    = float(trigger_s) if trigger_s else None,
                wochentage   = wochentage,
                aktionen     = data.get("aktionen", []),
                aktiv        = bool(data.get("aktiv", True)),
            )
        except Exception as e:
            log.error("Fehler beim Laden von %s: %s", pfad.name, e)
            return None

    async def _registriere_in_scheduler(self, routine: Routine) -> None:
        """Registriert eine Routine im bestehenden Scheduler."""
        if self._scheduler is None:
            return
        if not routine.aktiv:
            return

        fabrik: Callable[[], Coroutine] | None = None

        if routine.trigger_typ == "uhrzeit" and routine.trigger_zeit:
            # Wochentags-Filter einbauen
            async def macher(r: Routine = routine) -> None:
                if r.wochentage:
                    from datetime import datetime
                    heute = datetime.now().weekday()  # 0=Mo
                    tag_name = ["mo", "di", "mi", "do", "fr", "sa", "so"][heute]
                    if tag_name not in r.wochentage:
                        return
                await self.ausfuehren(r.name)

            self._scheduler.registriere(
                name=f"routine_{routine.name}",
                fabrik=lambda r=routine: macher(r),
                uhrzeit=routine.trigger_zeit,
            )

        elif routine.trigger_typ == "intervall" and routine.trigger_s:
            self._scheduler.registriere(
                name=f"routine_{routine.name}",
                fabrik=lambda r=routine: self.ausfuehren(r.name),
                intervall_s=routine.trigger_s,
            )
        # "manuell" → kein automatischer Trigger

    async def ausfuehren(self, name: str) -> list[str]:
        """Führt eine Routine manuell aus. Gibt Ergebnis-Liste zurück."""
        routine = self._routinen.get(name)
        if not routine:
            log.warning("Routine nicht gefunden: %s", name)
            return [f"Routine '{name}' nicht gefunden"]

        if not routine.aktiv:
            return [f"Routine '{name}' ist deaktiviert"]

        log.info("Routine startet: %s", name)
        ergebnisse: list[str] = []

        for aktion in routine.aktionen:
            try:
                ergebnis = await self._aktion_ausfuehren(aktion)
                ergebnisse.append(ergebnis)
            except Exception as e:
                ergebnisse.append(f"Fehler: {e}")
                log.error("Routine-Aktion Fehler (%s): %s", name, e)

        from datetime import datetime, timezone
        routine.letzter_run = datetime.now(timezone.utc).isoformat()
        log.info("Routine abgeschlossen: %s (%d Aktionen)", name, len(ergebnisse))
        return ergebnisse

    async def _aktion_ausfuehren(self, aktion: dict) -> str:
        """Führt eine einzelne Aktion aus."""
        typ = aktion.get("typ", "")

        if typ == "skill":
            return await self._aktion_skill(aktion)
        elif typ == "briefing":
            return await self._aktion_briefing()
        elif typ == "tts":
            return await self._aktion_tts(aktion)
        elif typ == "app_starten":
            return self._aktion_app(aktion)
        elif typ == "apex":
            return await self._aktion_apex(aktion)
        elif typ == "warten":
            await asyncio.sleep(float(aktion.get("sekunden", 1)))
            return "Gewartet"
        else:
            return f"Unbekannter Aktions-Typ: {typ}"

    async def _aktion_skill(self, aktion: dict) -> str:
        """Führt einen Skill aus."""
        from web.state import st
        from core.skill_registry import SkillContext, Hook
        skill_name = aktion.get("skill", "")
        user_input = aktion.get("user_input", skill_name)

        if not hasattr(st, "registry"):
            return "Skill-Registry nicht verfügbar"

        regs = st.registry.get_skill(skill_name)
        on_msg = [r for r in regs if r.hook == Hook.ON_MESSAGE and r.aktiv]
        if not on_msg:
            return f"Skill '{skill_name}' nicht gefunden"

        ctx = SkillContext(
            user_input=user_input,
            keywords=tuple(user_input.lower().split()[:5]),
            intents=(), brain_hits=(), session_facts=(),
            skill_config=on_msg[0].config, config={},
        )
        import inspect
        fn = on_msg[0].funktion
        if inspect.iscoroutinefunction(fn):
            result = await fn(ctx)
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, fn, ctx)

        return result.inhalt if result else f"Skill '{skill_name}' lieferte kein Ergebnis"

    async def _aktion_briefing(self) -> str:
        """Generiert Daily Briefing."""
        try:
            from web.state import st
            if hasattr(st, "briefing"):
                result = await st.briefing.generiere()
                return str(result)[:500]
        except Exception as e:
            return f"Briefing-Fehler: {e}"
        return "Briefing nicht verfügbar"

    async def _aktion_tts(self, aktion: dict) -> str:
        """Spricht Text via TTS."""
        text = aktion.get("text", "")
        try:
            from layer6.voice import VoiceIO
            voice = VoiceIO()
            await voice.spreche(text)
            return f"TTS: {text[:50]}"
        except Exception as e:
            return f"TTS nicht verfügbar: {e}"

    def _aktion_app(self, aktion: dict) -> str:
        """Startet eine App."""
        try:
            from layer6.computer_controller import ComputerController
            cc = ComputerController()
            return cc.app_starten(aktion)
        except Exception as e:
            return f"App-Start Fehler: {e}"

    async def _aktion_apex(self, aktion: dict) -> str:
        """Startet Apex-Agent."""
        if self._orch is None:
            return "Apex-Orchestrator nicht verfügbar"
        aufgabe = aktion.get("aufgabe", "")
        if not aufgabe:
            return "Keine Aufgabe für Apex angegeben"
        ergebnisse = []
        async for event in self._orch.starten(aufgabe, max_iter=20):
            if event.get("typ") == "fertig":
                ergebnisse.append(event.get("ergebnis", ""))
        return ergebnisse[-1] if ergebnisse else "Apex lieferte kein Ergebnis"

    def status(self) -> dict:
        return {
            "routinen": len(self._routinen),
            "aktiv": sum(1 for r in self._routinen.values() if r.aktiv),
            "namen": list(self._routinen.keys()),
        }

    def alle(self) -> list[dict]:
        return [
            {
                "name":         r.name,
                "trigger_typ":  r.trigger_typ,
                "trigger_zeit": r.trigger_zeit,
                "trigger_s":    r.trigger_s,
                "wochentage":   r.wochentage,
                "aktionen":     len(r.aktionen),
                "aktiv":        r.aktiv,
                "letzter_run":  r.letzter_run,
            }
            for r in self._routinen.values()
        ]
