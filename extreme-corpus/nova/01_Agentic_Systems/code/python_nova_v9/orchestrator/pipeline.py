"""Nova v9 – Pipeline.

Verarbeitet jede Nutzer-Nachricht und liefert alles was der Chat-Handler braucht.

v8-Problem geloest:
    v8 hatte 3 "parallele" Lanes, wobei Lane A aktiv auf Lane B wartete
    (busy-wait mit 8x50ms sleep). De facto sequenziell mit Overhead.

v9-Loesung:
    Ehrlich sequenziell, klar strukturiert, schnell.
    Schritt 1: Parse (sync, <1ms)
    Schritt 2: Embed (async, ~15ms)
    Schritt 3: Route (sync, <1ms, braucht Embedding)
    Schritt 4: Skills ausfuehren (async, parallel via SkillMesh)
    Kein verstecktes Warten, keine falschen Versprechen.

Ergebnis:
    PipelineErgebnis enthaelt alles was web/chat.py braucht:
    keywords, intents, aktive_skills, skill_ergebnisse, q_vec, todo_signal
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.logger import get
from core.skill_registry import SkillContext, SkillResult, Hook

if TYPE_CHECKING:
    from core.nano_parser      import NanoParser
    from core.ollama_client    import OllamaClient
    from core.router           import SemanticRouter
    from core.skill_registry   import SkillRegistry
    from orchestrator.skill_mesh import SkillMesh

log = get("pipeline")


@dataclass
class PipelineErgebnis:
    keywords:         list[str]              = field(default_factory=list)
    intents:          list[str]              = field(default_factory=list)
    q_vec:            list[float] | None     = None
    aktive_skills:    list[tuple[str, float]] = field(default_factory=list)
    skill_ergebnisse: dict[str, SkillResult] = field(default_factory=dict)
    todo_signal:      bool                   = False
    todo_text:        str                    = ""
    dauer_ms:         float                  = 0.0


class Pipeline:
    def __init__(
        self,
        parser:   "NanoParser",
        ollama:   "OllamaClient",
        router:   "SemanticRouter",
        mesh:     "SkillMesh",
        cfg:      dict,
    ) -> None:
        self._parser = parser
        self._ollama = ollama
        self._router = router
        self._mesh   = mesh
        self._cfg    = cfg

    async def verarbeite(self, user_input: str) -> PipelineErgebnis:
        """
        Verarbeitet eine Nutzer-Nachricht vollstaendig.
        Gibt PipelineErgebnis zurueck — bereit fuer den Chat-Handler.
        """
        start = time.monotonic()
        log.info(f"Pipeline START | '{user_input[:80]}'")

        ergebnis = PipelineErgebnis()
        loop = asyncio.get_running_loop()

        # ── Schritt 1: Parse (Keywords, Intents, Todo-Signal) ─────────────────
        parse = await loop.run_in_executor(None, self._parser.parse, user_input)
        ergebnis.keywords    = parse.keywords
        ergebnis.intents     = parse.intents
        ergebnis.todo_signal = parse.todo_signal
        ergebnis.todo_text   = parse.todo_text
        log.debug(f"Parse | keywords={parse.keywords} | intents={parse.intents} | todo={parse.todo_signal}")

        # ── Schritt 2: Embedding berechnen ────────────────────────────────────
        q_vec = await loop.run_in_executor(None, self._ollama.embed, user_input)
        ergebnis.q_vec = q_vec
        if q_vec:
            log.debug(f"Embedding | dim={len(q_vec)}")
        else:
            log.warning("Embedding fehlgeschlagen - Routing ohne Vektor")

        # ── Schritt 3: Skill-Routing ───────────────────────────────────────────
        if self._router.ist_bereit:
            routen = await loop.run_in_executor(
                None, self._router.route, user_input, q_vec
            )
            ergebnis.aktive_skills = routen
            if routen:
                log.debug(f"Routing | {[f'{n}={s}' for n, s in routen]}")
            else:
                log.debug("Routing | kein Skill aktiviert -> direkter LLM-Chat")
        else:
            log.warning("Router nicht bereit")

        # ── Schritt 4: Skills ausfuehren (parallel via SkillMesh) ─────────────
        if ergebnis.aktive_skills:
            skill_namen = [name for name, _ in ergebnis.aktive_skills]
            log.info(f"Skills | {skill_namen}")
            ctx = SkillContext(
                user_input = user_input,
                keywords   = ergebnis.keywords,
                intents    = ergebnis.intents,
                config     = self._cfg,
            )
            ergebnis.skill_ergebnisse = await self._mesh.ausfuehren(
                Hook.ON_MESSAGE, skill_namen, ctx, self._cfg
            )
            log.debug(f"Skill-Ergebnisse | {list(ergebnis.skill_ergebnisse.keys())}")

        ergebnis.dauer_ms = round((time.monotonic() - start) * 1000, 1)
        log.info(
            f"Pipeline FERTIG | {ergebnis.dauer_ms}ms | "
            f"skills={[n for n, _ in ergebnis.aktive_skills]}"
        )
        return ergebnis
