"""Nova v8 - Zentrales Logging.

Format: HH:MM:SS.mmm  LEVEL    nova.modul               - Nachricht

Nutzung:
    from core.logger import get
    log = get("brain")

Debug-Modus aktivieren (zeigt ALLES):
    Option 1 - Umgebungsvariable vor dem Start setzen:
        Windows:  set NOVA_DEBUG=1 && python -X utf8 main.py
        Linux:    NOVA_DEBUG=1 python -X utf8 main.py

    Option 2 - In config.yaml:
        logging:
          debug_modus: true

Was wird im Debug-Modus zusaetzlich geloggt:
    - Jeder Pipeline-Schritt mit Timing (Lane A/B/C)
    - NanoParser: Keywords, Intents, Todo-Signal
    - Ollama: Jede Anfrage mit Modell + Dauer
    - Brain: Jede Suche mit Treffer-Anzahl + Scores
    - Memory: WorkingMemory, EpisodicMemory, SemanticMemory, Coordinator
    - Skills: Routing-Entscheidung + Ausfuehrung + Ergebnis
    - EventBus: Alle Events
    - ContextManager: Token-Verbrauch pro Nachricht
"""
from __future__ import annotations

import logging
import sys
import time
import os
from pathlib import Path
from typing import Any

_LOG_FILE         = Path("nova.log")
_file_handler:    logging.FileHandler | None = None
_root_configured: bool = False
_debug_modus:     bool = False

_formatter = logging.Formatter(
    fmt="%(asctime)s.%(msecs)03d  %(levelname)-8s %(name)-20s - %(message)s",
    datefmt="%H:%M:%S",
)


def _ensure_configured() -> None:
    global _root_configured, _file_handler, _debug_modus

    if _root_configured:
        return

    # Debug-Modus: Env-Variable NOVA_DEBUG=1
    _debug_modus = os.environ.get("NOVA_DEBUG", "").strip() not in ("", "0", "false", "False")

    root = logging.getLogger("nova")
    root.setLevel(logging.DEBUG)  # Root immer DEBUG - Handler filtern selbst

    # === Stdout-Handler ===
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(_formatter)
    # Im Debug-Modus: alles auf Stdout. Normal: nur INFO+
    sh.setLevel(logging.DEBUG if _debug_modus else logging.INFO)
    root.addHandler(sh)

    # === FileHandler - IMMER DEBUG (vollstaendiges Log fuer Fehlersuche) ===
    try:
        fh = logging.FileHandler(_LOG_FILE, encoding="utf-8", mode="a")
        fh.setFormatter(_formatter)
        fh.setLevel(logging.DEBUG)  # Datei bekommt immer alles
        root.addHandler(fh)
        _file_handler = fh
    except PermissionError:
        root.warning("nova.log gesperrt - nur stdout-Logging aktiv")
    except Exception as e:
        root.warning(f"nova.log nicht erstellbar: {e}")

    root.propagate = False
    _root_configured = True

    lv = "DEBUG (alles)" if _debug_modus else "INFO (normal) - Datei: immer DEBUG"
    root.info(f"Logger bereit | Stdout-Level: {lv} | Log: {_LOG_FILE.resolve()}")
    if not _debug_modus:
        root.info("Tipp: Debug-Modus mit  set NOVA_DEBUG=1  oder  logging.debug_modus: true  in config.yaml")


def get(modul: str) -> logging.Logger:
    """Gibt Logger 'nova.<modul>' zurueck."""
    _ensure_configured()
    return logging.getLogger(f"nova.{modul}")


def set_level(level: int) -> None:
    """Aendert stdout-Level fuer alle nova.*-Logger."""
    root = logging.getLogger("nova")
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(level)


def aktiviere_debug() -> None:
    """Schaltet Debug-Modus zur Laufzeit ein (Stdout + Datei)."""
    global _debug_modus
    _ensure_configured()
    _debug_modus = True
    root = logging.getLogger("nova")
    root.setLevel(logging.DEBUG)
    for h in root.handlers:
        h.setLevel(logging.DEBUG)
    get("logger").info(">>> Debug-Modus aktiviert - ab jetzt wird alles geloggt <<<")


def ist_debug() -> bool:
    """True wenn Debug-Modus aktiv."""
    _ensure_configured()
    return _debug_modus


def lade_config_level(cfg: dict) -> None:
    """Liest Logging-Einstellungen aus config.yaml (nach dem Start aufrufen)."""
    _ensure_configured()
    debug = cfg.get("logging", {}).get("debug_modus", False)
    if debug and not _debug_modus:
        aktiviere_debug()


# ─── Timer-Hilfsklasse ───────────────────────────────────────────────────────

class Timer:
    """Kontextmanager fuer Zeitmessungen.

    Beispiel:
        with Timer(log, "Brain-Suche") as t:
            ergebnis = brain.suche(...)
        # Loggt automatisch: "Brain-Suche: 42.3ms"
    """

    def __init__(self, logger: logging.Logger, name: str,
                 level: int = logging.DEBUG) -> None:
        self._log   = logger
        self._name  = name
        self._level = level
        self._start = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type: Any, *_: Any) -> None:
        dauer_ms = round((time.monotonic() - self._start) * 1000, 1)
        if exc_type is not None:
            self._log.error(f"{self._name} FEHLER nach {dauer_ms}ms")
        else:
            self._log.log(self._level, f"{self._name}: {dauer_ms}ms")

    @property
    def ms(self) -> float:
        """Vergangene Millisekunden seit Start (auch waehrend Ausfuehrung nutzbar)."""
        return round((time.monotonic() - self._start) * 1000, 1)
