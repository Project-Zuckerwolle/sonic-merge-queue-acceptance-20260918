"""Nova Predator v1 — SkillMesh.

Führt mehrere Skills parallel aus.
Python 3.14: asyncio.TaskGroup (stable seit 3.11) statt gather für bessere
Fehlerbehandlung. Jeder Skill läuft isoliert — ein Fehler stoppt nicht die anderen.

Max 3 parallele Skills (config: router.max_skills).
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
