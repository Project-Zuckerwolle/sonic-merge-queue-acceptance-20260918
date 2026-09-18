"""Nova Predator v1 — SkillLoader.
Lädt Skills aus dem skills/-Verzeichnis und überwacht den inbox/-Ordner.
Windows-kompatibles Polling (kein inotify/watchdog nötig).
Python 3.14: asyncio.to_thread() für Dateisystem-Operationen, keine deprecated APIs.
"""
from __future__ import annotations
import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from core.event_bus import EventBus, EventTyp
from core.logger import get
from core.skill_registry import Hook, SkillRegistrierung, SkillRegistry

if TYPE_CHECKING:
    from core.ollama_client import OllamaClient
    from core.router import SemanticRouter, SkillRoute

log = get("skill_loader")

_PFLICHT_YAML_FELDER = ("name", "hooks")


class SkillLoader:
    """Lädt und verwaltet Skills aus skills/ und inbox/."""

    def __init__(
        self,
        registry: SkillRegistry,
        router: "SemanticRouter",
        bus: EventBus,
        skills_pfad: str | Path = "skills",
        inbox_pfad: str | Path = "inbox",
        ollama: "OllamaClient | None" = None,
        embed_modell: str = "nomic-embed-text",
        poll_intervall_s: float = 3.0,
    ) -> None:
        self._registry = registry
        self._router = router
        self._bus = bus
        self._skills_pfad = Path(skills_pfad)
        self._inbox_pfad = Path(inbox_pfad)
        self._ollama = ollama
        self._embed_modell = embed_modell
        self._poll_intervall_s = poll_intervall_s
        self._bekannte_inbox: set[str] = set()
        self._inbox_task: asyncio.Task | None = None
        self._geladene_skills: set[str] = set()

    async def alle_laden(self) -> int:
        """Lädt alle Skills aus dem skills/-Verzeichnis. Gibt Anzahl zurück."""
        geladen = 0
        if not self._skills_pfad.exists():
            log.warning("Skills-Verzeichnis nicht gefunden: %s", self._skills_pfad)
            return 0

        for skill_dir in sorted(self._skills_pfad.iterdir()):
            if not skill_dir.is_dir():
                continue
            ok = await self._lade_skill(skill_dir)
            if ok:
                geladen += 1

        log.info("Skills geladen: %d", geladen)
        return geladen

    async def inbox_watcher_starten(self) -> None:
        """Startet den Inbox-Watcher (Windows-kompatibles Polling)."""
        if not self._inbox_pfad.exists():
            self._inbox_pfad.mkdir(parents=True, exist_ok=True)

        # Initialen Stand erfassen
        self._bekannte_inbox = await asyncio.to_thread(self._inbox_scan)
        self._inbox_task = asyncio.create_task(
            self._inbox_poll_loop(), name="inbox_watcher"
        )
        log.info("Inbox-Watcher gestartet: %s", self._inbox_pfad)

    async def inbox_watcher_stoppen(self) -> None:
        if self._inbox_task and not self._inbox_task.done():
            self._inbox_task.cancel()
            try:
                await self._inbox_task
            except asyncio.CancelledError:
                pass

    async def _inbox_poll_loop(self) -> None:
        """Polling-Loop — prüft inbox/ alle poll_intervall_s Sekunden."""
        while True:
            await asyncio.sleep(self._poll_intervall_s)
            try:
                aktuell = await asyncio.to_thread(self._inbox_scan)
                neu = aktuell - self._bekannte_inbox

                for name in neu:
                    skill_dir = self._inbox_pfad / name
                    if skill_dir.is_dir():
                        ok = await self._lade_skill(skill_dir)
                        if ok:
                            # Skill in skills/-Verzeichnis verschieben
                            ziel = self._skills_pfad / name
                            if not ziel.exists():
                                await asyncio.to_thread(
                                    skill_dir.rename, ziel
                                )
                            log.info("Inbox-Skill installiert: %s", name)

                self._bekannte_inbox = aktuell
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("Inbox-Watcher Fehler: %s", e)

    def _inbox_scan(self) -> set[str]:
        """Gibt Namen aller Unterverzeichnisse in inbox/ zurück."""
        if not self._inbox_pfad.exists():
            return set()
        return {
            d.name for d in self._inbox_pfad.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        }

    async def _lade_skill(self, skill_dir: Path) -> bool:
        """Lädt einen einzelnen Skill aus einem Verzeichnis."""
        yaml_pfad = skill_dir / "skill.yaml"
        py_pfad   = skill_dir / "skill.py"

        if not yaml_pfad.exists() or not py_pfad.exists():
            return False

        try:
            # YAML laden
            meta = await asyncio.to_thread(self._lese_yaml, yaml_pfad)
            if not self._yaml_gueltig(meta):
                log.warning("Ungültiges skill.yaml in %s", skill_dir.name)
                return False

            skill_name = meta["name"]

            # Python-Modul laden
            modul = await asyncio.to_thread(self._lade_modul, py_pfad, skill_name)
            if modul is None:
                return False

            # Hooks registrieren
            hooks_registriert = 0
            for hook_cfg in meta.get("hooks", []):
                hook_name = hook_cfg.get("name", "")
                try:
                    hook = Hook(hook_name)
                except ValueError:
                    log.warning("Unbekannter Hook '%s' in %s", hook_name, skill_name)
                    continue

                fn = getattr(modul, hook_name, None)
                if fn is None:
                    continue

                threshold = float(hook_cfg.get("threshold", 0.74))
                beispiele = hook_cfg.get("beispiele", [])
                config_defaults = meta.get("config_defaults", {})

                reg = SkillRegistrierung(
                    name=skill_name,
                    hook=hook,
                    threshold=threshold,
                    funktion=fn,
                    config=config_defaults,
                )
                self._registry.registriere(reg)

                # Router-Einträge mit Beispiel-Embeddings
                if hook == Hook.ON_MESSAGE and beispiele and self._ollama:
                    await self._router_eintraege_erstellen(
                        skill_name, threshold, beispiele
                    )

                hooks_registriert += 1

            if hooks_registriert > 0:
                self._geladene_skills.add(skill_name)
                await self._bus.publish(
                    EventTyp.SKILL_INSTALLIERT,
                    {"name": skill_name, "hooks": hooks_registriert},
                )
                log.debug("Skill geladen: %s (%d Hooks)", skill_name, hooks_registriert)
                return True

        except Exception as e:
            log.error("Skill-Ladefehler in %s: %s", skill_dir.name, e)
            await self._bus.publish(
                EventTyp.SKILL_FEHLER,
                {"name": skill_dir.name, "fehler": str(e)},
            )
        return False

    async def _router_eintraege_erstellen(
        self,
        skill_name: str,
        threshold: float,
        beispiele: list[str],
    ) -> None:
        """Erstellt Router-Einträge mit Embeddings für Beispiel-Sätze."""
        from core.router import SkillRoute
        vektoren: list[list[float]] = []
        for beispiel in beispiele[:8]:   # Max 8 Beispiele
            try:
                v = await self._ollama.embed(beispiel, modell=self._embed_modell)
                if v:
                    vektoren.append(v)
            except Exception as e:
                log.debug("Embedding für Beispiel fehlgeschlagen: %s", e)

        if vektoren:
            self._router.route_registrieren(SkillRoute(
                name=skill_name,
                threshold=threshold,
                beispiele=beispiele,
                vektoren=vektoren,
            ))

    def _lese_yaml(self, pfad: Path) -> dict:
        with open(pfad, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _yaml_gueltig(self, meta: dict) -> bool:
        return all(f in meta for f in _PFLICHT_YAML_FELDER)

    def _lade_modul(self, py_pfad: Path, name: str):
        """Lädt Python-Modul dynamisch."""
        modul_name = f"nova_skill_{name.lower().replace(' ', '_')}"
        # Bereits geladen? Neu laden für Hot-Reload
        if modul_name in sys.modules:
            del sys.modules[modul_name]

        spec = importlib.util.spec_from_file_location(modul_name, py_pfad)
        if spec is None or spec.loader is None:
            return None
        modul = importlib.util.module_from_spec(spec)
        sys.modules[modul_name] = modul
        spec.loader.exec_module(modul)   # type: ignore[union-attr]
        return modul

    def geladene_skills(self) -> list[str]:
        return sorted(self._geladene_skills)
