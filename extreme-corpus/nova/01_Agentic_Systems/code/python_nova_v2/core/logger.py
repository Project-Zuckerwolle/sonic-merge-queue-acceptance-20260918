"""Nova Predator v1 — Logger.
Zentrales Logging-Setup. Kein Nova-Import.
Python 3.14 kompatibel.
"""
from __future__ import annotations
import logging
import os
import sys
from pathlib import Path


_LOGGERS: dict[str, logging.Logger] = {}
_INITIALISIERT = False


def initialisiere(debug: bool = False, logdatei: str = "nova.log") -> None:
    global _INITIALISIERT
    if _INITIALISIERT:
        return
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    # Root-Handler: Datei immer DEBUG
    datei_handler = logging.FileHandler(logdatei, encoding="utf-8")
    datei_handler.setLevel(logging.DEBUG)
    datei_handler.setFormatter(logging.Formatter(fmt, datefmt))

    # Konsole: abhängig von debug_modus
    konsole_handler = logging.StreamHandler(sys.stdout)
    konsole_handler.setLevel(level)
    konsole_handler.setFormatter(logging.Formatter(fmt, datefmt))

    root = logging.getLogger("nova")
    root.setLevel(logging.DEBUG)
    root.addHandler(datei_handler)
    root.addHandler(konsole_handler)
    root.propagate = False
    _INITIALISIERT = True


def get(name: str) -> logging.Logger:
    """Gibt einen benannten Nova-Logger zurück."""
    voll = f"nova.{name}"
    if voll not in _LOGGERS:
        _LOGGERS[voll] = logging.getLogger(voll)
    return _LOGGERS[voll]


def set_level(debug: bool) -> None:
    """Ändert Konsolen-Level zur Laufzeit."""
    for logger in _LOGGERS.values():
        for h in logging.getLogger("nova").handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.setLevel(logging.DEBUG if debug else logging.INFO)
