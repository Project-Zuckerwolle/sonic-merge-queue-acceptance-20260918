"""Nova v9 – SkillMesh.

Parallele Skill-Ausfuehrung mit Chaining-Support.
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from core.logger import get
from core.skill_registry import SkillContext, SkillResult, Hook

if TYPE_CHECKING:
    from core.skill_registry import SkillRegistry

log = get("skill_mesh")

MAX_CHAIN_TIEFE = 3


class SkillMesh:
    def __init__(self, registry: "SkillRegistry") -> None:
        self._registry = registry
        self._ausfuehrungen: list[dict] = []

    async def ausfuehren(
        self,
        hook_name:     str,
        aktive_skills: list[str],
        ctx:           SkillContext,
        cfg:           dict,
        tiefe:         int = 0,
    ) -> dict[str, SkillResult]:
        """Fuehrt alle aktiven Skills parallel aus, verarbeitet Chains."""
        if tiefe >= MAX_CHAIN_TIEFE:
            log.warning(f"Max Chain-Tiefe {MAX_CHAIN_TIEFE} erreicht")
            return {}

        if not aktive_skills:
            return {}

        start = time.monotonic()
        log.debug(f"SkillMesh | Tiefe={tiefe} | Skills={aktive_skills}")

        tasks = [
            self._skill_task(name, hook_name, ctx, cfg)
            for name in aktive_skills
        ]
        ergebnis_liste = await asyncio.gather(*tasks, return_exceptions=True)

        ergebnisse:  dict[str, SkillResult] = {}
        folge_skills: list[str] = []

        for name, ergebnis in zip(aktive_skills, ergebnis_liste):
            if isinstance(ergebnis, Exception):
                log.error(f"SkillMesh Fehler {name}: {ergebnis}", exc_info=True)
                continue
            if ergebnis is not None:
                ergebnisse[name] = ergebnis
                # Skill-Chaining
                if ergebnis.metadaten.get("chain_skills"):
                    for folge in ergebnis.metadaten["chain_skills"]:
                        if folge not in aktive_skills and folge in self._registry:
                            folge_skills.append(folge)

        if folge_skills:
            log.info(f"SkillMesh Chain: {folge_skills} (Tiefe {tiefe + 1})")
            folge_ergebnisse = await self.ausfuehren(
                hook_name, list(set(folge_skills)), ctx, cfg, tiefe + 1
            )
            ergebnisse.update(folge_ergebnisse)

        dauer = round(time.monotonic() - start, 3)
        self._ausfuehrungen.append({
            "skills":     aktive_skills,
            "ergebnisse": len(ergebnisse),
            "dauer_s":    dauer,
            "tiefe":      tiefe,
        })
        log.debug(f"SkillMesh fertig | {aktive_skills} -> {len(ergebnisse)} Ergebnisse | {dauer}s")
        return ergebnisse

    async def _skill_task(
        self,
        skill_name: str,
        hook_name:  str,
        ctx:        SkillContext,
        cfg:        dict,
    ) -> SkillResult | None:
        """Einzelnen Skill isoliert ausfuehren."""
        ergebnisse = await loop.run_in_executor(
            None,
            self._registry.execute_hook,
            hook_name,
            [skill_name],
            ctx,
            cfg,
        )
        return ergebnisse.get(skill_name)

    def letzter_lauf(self) -> dict | None:
        return self._ausfuehrungen[-1] if self._ausfuehrungen else None

    def stats(self) -> dict:
        if not self._ausfuehrungen:
            return {"laeufe": 0}
        return {
            "laeufe":         len(self._ausfuehrungen),
            "letzte_dauer_s": self._ausfuehrungen[-1]["dauer_s"],
        }
