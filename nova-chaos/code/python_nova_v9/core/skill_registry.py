"""Nova v8 – Skill Registry.

Hook-System für das Skill-Mesh.
Alle 5 Hook-Typen: ON_MESSAGE, ON_RESPONSE, ON_BRAIN_WRITE, ON_SESSION_END, ON_STARTUP.
Thread-sicher, copy-safe (v7 Bug: ctx-Mutation im Loop gefixt).
"""
from __future__ import annotations
import copy
import importlib.util
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from core.logger import get

log = get("registry")


# ─── Hook-Typen ──────────────────────────────────────────────────────────────
class Hook:
    ON_MESSAGE     = "on_message"      # Jede Chat-Nachricht
    ON_RESPONSE    = "on_response"     # Nach LLM-Antwort
    ON_BRAIN_WRITE = "on_brain_write"  # Wenn Brain-Entry geschrieben wird
    ON_SESSION_END = "on_session_end"  # Session wird archiviert
    ON_STARTUP     = "on_startup"      # Server-Start

    ALLE = [ON_MESSAGE, ON_RESPONSE, ON_BRAIN_WRITE, ON_SESSION_END, ON_STARTUP]


# ─── SkillContext ─────────────────────────────────────────────────────────────
@dataclass
class SkillContext:
    user_input:    str
    keywords:      list[str]          = field(default_factory=list)
    intents:       list[str]          = field(default_factory=list)
    brain_hits:    list[dict]         = field(default_factory=list)
    session:       list[dict]         = field(default_factory=list)
    config:        dict               = field(default_factory=dict)
    skill_config:  dict               = field(default_factory=dict)
    # Extras für ON_RESPONSE, ON_BRAIN_WRITE etc.
    antwort:       str                = ""
    brain_entry:   dict | None        = None
    working_memory: Any               = None   # WorkingMemory-Instanz
    event_bus:     Any                = None   # EventBus-Instanz


# ─── SkillResult ─────────────────────────────────────────────────────────────
@dataclass
class SkillResult:
    inhalt:       str
    typ:          str   = "info"    # info | aktion | datei | stream
    ui_nachricht: str   = ""        # Direktnachricht an UI (bypassed LLM)
    weiter_aktiv: bool  = False     # True → oranges Blinken (z.B. Claw läuft)
    metadaten:    dict  = field(default_factory=dict)
    skill_name:   str   = ""


# ─── Skill-Registrierung ─────────────────────────────────────────────────────
@dataclass
class SkillRegistrierung:
    name:          str
    version:       str
    beschreibung:  str
    pfad:          Path
    hooks:         dict[str, Callable]   = field(default_factory=dict)
    config_defaults: dict                = field(default_factory=dict)
    beispiele:     list[str]             = field(default_factory=list)
    threshold:     float                 = 0.63
    aktiv:         bool                  = True


# ─── Skill Registry ───────────────────────────────────────────────────────────
class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillRegistrierung] = {}

    def registriere(
        self,
        name: str,
        version: str,
        beschreibung: str,
        pfad: Path,
        hooks: dict[str, Callable],
        config_defaults: dict | None = None,
        beispiele: list[str] | None = None,
        threshold: float = 0.63,
    ) -> None:
        self._skills[name] = SkillRegistrierung(
            name=name,
            version=version,
            beschreibung=beschreibung,
            pfad=pfad,
            hooks=hooks,
            config_defaults=config_defaults or {},
            beispiele=beispiele or [],
            threshold=threshold,
        )
        log.info(f"Skill registriert: {name} v{version} (Hooks: {list(hooks.keys())})")

    def deregistriere(self, name: str) -> None:
        self._skills.pop(name, None)
        log.info(f"Skill entfernt: {name}")

    def skill_beispiele(self) -> dict[str, dict]:
        """Für SemanticRouter: {name: {beispiele, threshold}}"""
        return {
            name: {"beispiele": s.beispiele, "threshold": s.threshold}
            for name, s in self._skills.items()
            if s.aktiv and s.beispiele
        }

    # ─── Hook ausführen ──────────────────────────────────────────────────────
    def execute_hook(
        self,
        hook_name: str,
        aktive_skills: list[str],
        ctx: SkillContext,
        global_cfg: dict,
    ) -> dict[str, SkillResult]:
        """
        Führt Hook für alle aktiven Skills aus.
        Gibt {skill_name: SkillResult} zurück.
        Sicher: copy.copy(ctx) pro Skill (v7 Bug-Fix).
        """
        ergebnisse: dict[str, SkillResult] = {}

        for skill_name in aktive_skills:
            reg = self._skills.get(skill_name)
            if not reg or not reg.aktiv:
                continue
            hook_fn = reg.hooks.get(hook_name)
            if not hook_fn:
                continue

            # Eigene Kopie pro Skill (verhindert ctx-Mutation zwischen Skills)
            ctx_kopie = copy.copy(ctx)
            ctx_kopie.skill_config = {
                **reg.config_defaults,
                **global_cfg.get("skills", {}).get(skill_name, {}),
            }

            try:
                result = hook_fn(ctx_kopie)
                if result is not None:
                    result.skill_name = skill_name
                    ergebnisse[skill_name] = result
            except Exception as e:
                log.error(f"Hook-Fehler {skill_name}.{hook_name}: {e}", exc_info=True)

        return ergebnisse

    # ─── Skill laden aus Datei ───────────────────────────────────────────────
    def lade_modul(self, pfad: Path, modul_name: str) -> Any | None:
        """Lädt skill.py dynamisch."""
        try:
            spec = importlib.util.spec_from_file_location(modul_name, pfad)
            if spec is None or spec.loader is None:
                return None
            modul = importlib.util.module_from_spec(spec)
            sys.modules[modul_name] = modul
            spec.loader.exec_module(modul)
            return modul
        except Exception as e:
            log.error(f"Modul-Ladefehler {pfad}: {e}", exc_info=True)
            return None

    # ─── Info ─────────────────────────────────────────────────────────────────
    def alle_skills(self) -> list[dict]:
        return [
            {
                "name": s.name,
                "version": s.version,
                "beschreibung": s.beschreibung,
                "hooks": list(s.hooks.keys()),
                "aktiv": s.aktiv,
                "threshold": s.threshold,
                "pfad": str(s.pfad),
            }
            for s in self._skills.values()
        ]

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills
