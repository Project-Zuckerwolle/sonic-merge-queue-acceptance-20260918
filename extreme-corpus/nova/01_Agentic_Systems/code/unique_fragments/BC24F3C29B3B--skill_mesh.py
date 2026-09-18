"""Nova Predator v3 — SkillMesh.

Führt mehrere Skills parallel aus.
Python 3.14: asyncio.TaskGroup (stable seit 3.11) statt gather für bessere
Fehlerbehandlung. Jeder Skill läuft isoliert — ein Fehler stoppt nicht die anderen.

Max 3 parallele Skills (config: router.max_skills).

v3-Änderungen (additiv):
  - alle_tool_definitionen(): gibt Tool-Definitionen aller aktiven Skills zurück
  - tool_aufrufen(): führt einen einzelnen Tool-Call aus (für Tool-Loop in nova_ws)
"""
from __future__ import annotations
import asyncio
import inspect
from typing import Any

from core.logger import get
from core.skill_registry import (
    Hook,
    SkillContext,
    SkillRegistrierung,
    SkillRegistry,
    SkillResult,
)

log = get("skill_mesh")


class SkillMesh:
    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    async def ausfuehren(
        self,
        skill_namen: list[str],
        user_input: str,
        keywords: tuple[str, ...],
        intents: tuple[str, ...],
        q_vec: list[float],
        config: dict[str, Any],
        brain_hits: tuple[Any, ...] = (),
        session_facts: tuple[Any, ...] = (),
    ) -> dict[str, SkillResult]:
        """Führt alle angegebenen Skills parallel aus.

        Jeder Skill bekommt einen eigenen SkillContext (frozen).
        Fehler in einzelnen Skills werden geloggt, stoppen aber nicht die anderen.
        """
        if not skill_namen:
            return {}

        # Registrierungen für aktive Skills holen
        aufgaben: list[tuple[str, SkillRegistrierung]] = []
        for name in skill_namen:
            regs = self._registry.get_skill(name)
            on_msg = [r for r in regs if r.hook == Hook.ON_MESSAGE and r.aktiv]
            if on_msg:
                aufgaben.append((name, on_msg[0]))

        if not aufgaben:
            return {}

        ergebnisse: dict[str, SkillResult] = {}

        async def _skill_task(name: str, reg: SkillRegistrierung) -> None:
            ctx = SkillContext(
                user_input=user_input,
                keywords=keywords,
                intents=intents,
                brain_hits=brain_hits,
                session_facts=session_facts,
                skill_config=reg.config,
                config=config,
            )
            try:
                fn = reg.funktion
                # Python 3.14: inspect.iscoroutinefunction statt asyncio.iscoroutinefunction
                if inspect.iscoroutinefunction(fn):
                    result = await fn(ctx)
                else:
                    # Sync-Funktion in Executor ausführen
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, fn, ctx)

                if result is not None:
                    ergebnisse[name] = result
                    log.debug("Skill '%s' OK: %d Zeichen", name, len(result.inhalt))
            except Exception as e:
                log.error("Skill '%s' Fehler: %s", name, e)

        # Parallel ausführen — Python 3.14 TaskGroup für saubere Fehlerbehandlung
        # TaskGroup bricht bei erstem Fehler ab, daher wrappen wir Fehler intern
        async with asyncio.TaskGroup() as tg:
            for name, reg in aufgaben:
                tg.create_task(_skill_task(name, reg), name=f"skill_{name}")

        return ergebnisse

    # ── Tool-Calling API (v3, additiv) ───────────────────────────────────────

    def alle_tool_definitionen(self) -> list[dict]:
        """Gibt Tool-Definitionen aller aktiven ON_MESSAGE Skills zurück.

        Jeder Skill der eine tool_definition()-Funktion exportiert wird
        einbezogen. Skills ohne tool_definition() werden übersprungen.

        Returns:
            Liste von Tool-Definitionen im Format:
            [{"type": "function", "function": {"name": ..., ...}}, ...]
        """
        tools: list[dict] = []
        for name in self._registry.alle_namen():
            regs = self._registry.get_skill(name)
            aktiv = [r for r in regs if r.hook == Hook.ON_MESSAGE and r.aktiv]
            if not aktiv:
                continue
            # tool_definition() über das Modul der Funktion holen
            fn = aktiv[0].funktion
            modul = getattr(fn, "__module__", None)
            if modul:
                import sys
                m = sys.modules.get(modul)
                if m and hasattr(m, "tool_definition"):
                    try:
                        td = m.tool_definition()
                        tools.append(td)
                        log.debug("Tool-Definition geladen: %s", name)
                    except Exception as e:
                        log.warning("tool_definition() Fehler für '%s': %s", name, e)
        return tools

    async def tool_aufrufen(
        self,
        tool_name: str,
        argumente: dict,
        user_input: str = "",
    ) -> str:
        """Führt einen einzelnen Tool-Call aus (für Tool-Loop in nova_ws).

        Baut einen SkillContext aus den Tool-Argumenten und ruft on_message() auf.
        Die Argumente werden sowohl als skill_config als auch im user_input
        übergeben, damit bestehende Skills (die ctx.user_input nutzen) funktionieren.

        Args:
            tool_name:  Name des Skills (muss registriert sein)
            argumente:  Vom Modell übergebene Argumente
            user_input: Original-User-Input (für Skills die ctx.user_input nutzen)

        Returns:
            Ergebnis-String oder Fehlermeldung
        """
        regs = self._registry.get_skill(tool_name)
        aktiv = [r for r in regs if r.hook == Hook.ON_MESSAGE and r.aktiv]

        if not aktiv:
            log.warning("tool_aufrufen: Skill '%s' nicht gefunden oder inaktiv", tool_name)
            return f"Tool '{tool_name}' nicht verfügbar."

        reg = aktiv[0]

        # Argumente in skill_config einbauen (überschreibt defaults)
        merged_config = {**reg.config, **argumente}

        # user_input anreichern wenn Argumente vorhanden (für Skills die
        # ctx.user_input parsen statt skill_config nutzen)
        effektiver_input = user_input
        if argumente and not effektiver_input:
            # Fallback: Argumente als natürlichen Text formulieren
            effektiver_input = " ".join(str(v) for v in argumente.values())

        ctx = SkillContext(
            user_input=effektiver_input,
            keywords=tuple(str(v) for v in argumente.values()),
            intents=(),
            brain_hits=(),
            session_facts=(),
            skill_config=merged_config,
            config={},
        )

        try:
            fn = reg.funktion
            if inspect.iscoroutinefunction(fn):
                result = await fn(ctx)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, fn, ctx)

            if result is None:
                return f"Tool '{tool_name}' hat kein Ergebnis zurückgegeben."

            log.debug("tool_aufrufen '%s' OK: %d Zeichen", tool_name, len(result.inhalt))
            return result.inhalt

        except Exception as e:
            log.error("tool_aufrufen '%s' Fehler: %s", tool_name, e)
            return f"Fehler bei Tool '{tool_name}': {e}"

    async def startup_hooks(self, config: dict[str, Any]) -> None:
        """Führt on_startup Hooks aller aktiven Skills aus."""
        regs = self._registry.get_hooks(Hook.ON_STARTUP)
        if not regs:
            return

        for reg in regs:
            ctx = SkillContext(
                user_input="",
                keywords=(),
                intents=(),
                brain_hits=(),
                session_facts=(),
                skill_config=reg.config,
                config=config,
            )
            try:
                fn = reg.funktion
                if inspect.iscoroutinefunction(fn):
                    await fn(ctx)
                else:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, fn, ctx)
                log.debug("Startup-Hook '%s' OK", reg.name)
            except Exception as e:
                log.warning("Startup-Hook '%s' Fehler: %s", reg.name, e)
