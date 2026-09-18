"""Nova v8 – SkillLoader.

Überwacht inbox/, installiert Skills automatisch.
Unterstützt ZIP-Dateien und Ordner.
Hot-Reload: Router-Embeddings werden nach Installation aktualisiert.
"""
from __future__ import annotations
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from core.logger import get
from core.event_bus import bus, EventTyp

if TYPE_CHECKING:
    from core.skill_registry  import SkillRegistry
    from core.router import SemanticRouter

log = get("loader")

PFLICHT_FELDER = {"name", "version", "beschreibung"}


class SkillLoader:
    def __init__(
        self,
        skills_pfad: str,
        inbox_pfad:  str,
        registry:    "SkillRegistry",
        router:      "SemanticRouter",
        cfg:         dict,
    ) -> None:
        self._skills_p = Path(skills_pfad)
        self._inbox_p  = Path(inbox_pfad)
        self._failed_p = self._inbox_p / "_failed"
        self._registry = registry
        self._router   = router
        self._cfg      = cfg
        self._watcher_thread: threading.Thread | None = None
        self._lauft = False

        self._skills_p.mkdir(parents=True, exist_ok=True)
        self._inbox_p.mkdir(parents=True, exist_ok=True)
        self._failed_p.mkdir(parents=True, exist_ok=True)

    # ─── Startup: alle Skills laden ──────────────────────────────────────────
    def lade_alle(self) -> int:
        geladen = 0
        for skill_dir in sorted(self._skills_p.iterdir()):
            if not skill_dir.is_dir() or skill_dir.name.startswith("_"):
                continue
            if self._lade_skill(skill_dir):
                geladen += 1
        log.info(f"Skills geladen: {geladen}")
        return geladen

    # ─── Inbox-Watcher ───────────────────────────────────────────────────────
    def starte_watcher(self) -> None:
        self._lauft = True
        self._watcher_thread = threading.Thread(
            target=self._watch_loop, daemon=True, name="skill_watcher"
        )
        self._watcher_thread.start()
        log.info("Inbox-Watcher gestartet")

    def stoppe_watcher(self) -> None:
        self._lauft = False

    def _watch_loop(self) -> None:
        while self._lauft:
            try:
                for item in list(self._inbox_p.iterdir()):
                    if item.name.startswith("_"):
                        continue
                    if item.suffix.lower() == ".zip":
                        self._installiere_zip(item)
                    elif item.is_dir():
                        self._installiere_ordner(item)
            except Exception as e:
                log.error(f"Watcher-Fehler: {e}")
            time.sleep(3)

    # ─── Installation ─────────────────────────────────────────────────────────
    def _installiere_zip(self, zip_pfad: Path) -> None:
        name = zip_pfad.stem
        try:
            with zipfile.ZipFile(zip_pfad) as zf:
                zf.extractall(self._inbox_p / name)
            zip_pfad.unlink()
            self._installiere_ordner(self._inbox_p / name)
        except Exception as e:
            self._fehler(zip_pfad, str(e))

    def _installiere_ordner(self, ordner: Path) -> None:
        ziel = self._skills_p / ordner.name
        try:
            self._validiere(ordner)
            if ziel.exists():
                shutil.rmtree(ziel)
            shutil.copytree(ordner, ziel)
            shutil.rmtree(ordner)

            self._installiere_deps(ziel)
            if self._lade_skill(ziel):
                name = self._hole_name(ziel)
                log.info(f"Skill installiert: {name}")
                bus.publish_threadsafe(EventTyp.SKILL_INSTALLIERT, {"name": name})
        except Exception as e:
            self._fehler(ordner, str(e))

    def _installiere_deps(self, skill_dir: Path) -> None:
        yaml_p = skill_dir / "skill.yaml"
        if not yaml_p.exists():
            return
        try:
            meta = yaml.safe_load(yaml_p.read_text(encoding="utf-8"))
            deps = meta.get("dependencies", [])
            if deps:
                import subprocess
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--break-system-packages", "-q"] + deps,
                    capture_output=True, timeout=60
                )
                if result.returncode != 0:
                    log.warning(f"pip install Warnung: {result.stderr.decode()[:200]}")
        except Exception as e:
            log.warning(f"Dep-Install-Fehler: {e}")

    # ─── Einzelnen Skill laden ────────────────────────────────────────────────
    def _lade_skill(self, skill_dir: Path) -> bool:
        yaml_p = skill_dir / "skill.yaml"
        py_p   = skill_dir / "skill.py"
        if not yaml_p.exists() or not py_p.exists():
            return False

        try:
            meta   = yaml.safe_load(yaml_p.read_text(encoding="utf-8"))
            name   = meta.get("name", skill_dir.name)
            modul  = self._registry.lade_modul(py_p, f"nova_skill_{skill_dir.name}")
            if modul is None:
                return False

            # Hook-Funktionen aus Modul extrahieren
            from core.skill_registry import Hook
            hooks = {}
            for hook_name in Hook.ALLE:
                fn = getattr(modul, hook_name, None)
                if callable(fn):
                    hooks[hook_name] = fn

            if not hooks:
                log.warning(f"Skill {name}: keine Hooks gefunden")
                return False

            # Beispiele aus hooks-Liste in YAML
            beispiele: list[str] = []
            threshold = 0.63
            for h in meta.get("hooks", []):
                if h.get("name") == "on_message":
                    beispiele = h.get("beispiele", [])
                    threshold = h.get("threshold", 0.63)
                    break

            self._registry.registriere(
                name=name,
                version=str(meta.get("version", "1.0")),
                beschreibung=meta.get("beschreibung", ""),
                pfad=skill_dir,
                hooks=hooks,
                config_defaults=meta.get("config_defaults", {}),
                beispiele=beispiele,
                threshold=threshold,
            )

            # Router aktualisieren
            if beispiele and self._router.ist_bereit:
                self._router.skill_hinzufuegen(name, beispiele, threshold)

            return True

        except Exception as e:
            log.error(f"Skill-Ladefehler {skill_dir.name}: {e}", exc_info=True)
            bus.publish_threadsafe(EventTyp.SKILL_FEHLER, {"name": skill_dir.name, "fehler": str(e)})
            return False

    # ─── Validierung ─────────────────────────────────────────────────────────
    def _validiere(self, skill_dir: Path) -> None:
        yaml_p = skill_dir / "skill.yaml"
        py_p   = skill_dir / "skill.py"
        if not yaml_p.exists():
            raise ValueError(f"skill.yaml fehlt in {skill_dir.name}")
        if not py_p.exists():
            raise ValueError(f"skill.py fehlt in {skill_dir.name}")
        meta = yaml.safe_load(yaml_p.read_text(encoding="utf-8"))
        fehlend = PFLICHT_FELDER - set(meta.keys())
        if fehlend:
            raise ValueError(f"Pflichtfelder fehlen: {fehlend}")
        # Syntax-Check
        import ast
        try:
            ast.parse(py_p.read_text(encoding="utf-8"))
        except SyntaxError as e:
            raise ValueError(f"Syntaxfehler in skill.py: {e}")

    def _hole_name(self, skill_dir: Path) -> str:
        yaml_p = skill_dir / "skill.yaml"
        if yaml_p.exists():
            meta = yaml.safe_load(yaml_p.read_text(encoding="utf-8"))
            return meta.get("name", skill_dir.name)
        return skill_dir.name

    def _fehler(self, pfad: Path, fehler: str) -> None:
        log.error(f"Skill-Installation fehlgeschlagen ({pfad.name}): {fehler}")
        ziel = self._failed_p / pfad.name
        try:
            if pfad.is_dir():
                shutil.copytree(pfad, ziel, dirs_exist_ok=True)
                shutil.rmtree(pfad)
            elif pfad.is_file():
                shutil.move(str(pfad), ziel)
            (self._failed_p / f"{pfad.name}_error.txt").write_text(fehler, encoding="utf-8")
        except Exception:
            pass
        bus.publish_threadsafe(EventTyp.SKILL_FEHLER, {"name": pfad.name, "fehler": fehler})

    def skill_deaktivieren(self, name: str) -> bool:
        skill_dir = self._skills_p / name.lower()
        if not skill_dir.exists():
            return False
        ziel = self._skills_p / "_deactivated" / name.lower()
        ziel.parent.mkdir(exist_ok=True)
        shutil.move(str(skill_dir), ziel)
        self._registry.deregistriere(name)
        self._router.skill_entfernen(name)
        bus.publish_threadsafe(EventTyp.SKILL_DEAKTIVIERT, {"name": name})
        return True
