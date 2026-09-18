"""Nova Predator v1 — Config.
YAML laden, deep-merge, get(). Kein Nova-Import.
Python 3.14: from __future__ annotations für lazy eval.
"""
from __future__ import annotations
import copy
from pathlib import Path
from typing import Any

import yaml

_CFG: dict[str, Any] = {}
_GELADEN = False


def _deep_merge(basis: dict, override: dict) -> dict:
    result = copy.deepcopy(basis)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


_STANDARD: dict[str, Any] = {
    "ollama": {"host": "http://localhost:11434", "timeout": 120},
    "modelle": {
        "chat": "gemma4:e4b",
        "brain": "qwen2.5:3b",
        "embed": "nomic-embed-text",
        "schnell": "gemma4:e4b",
    },
    "router": {"min_score": 0.72, "max_skills": 2},
    "context": {"max_tokens": 8000, "compress_schwelle": 0.85},
    "brain": {
        "vertrauen_schwelle": 0.5,
        "extraktor_min_score": 0.6,
        "thinker_intervall_s": 1800,
        "durchbruch_schwelle": 0.75,
    },
    "schlaf": {"aktiv": True, "timeout_s": 300, "keep_brain_llm": True},
    "web": {"host": "127.0.0.1", "port": 8000},
    "logging": {"debug_modus": False},
    "scheduler": {"briefing_zeit": "07:00", "thinker_intervall_h": 0.5},
}


def laden(pfad: str | Path = "config.yaml") -> dict[str, Any]:
    global _CFG, _GELADEN
    cfg = copy.deepcopy(_STANDARD)
    p = Path(pfad)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            nutzer = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, nutzer)
    _CFG = cfg
    _GELADEN = True
    return _CFG


def get(schluessel: str, standard: Any = None) -> Any:
    """Dot-notation: get('ollama.host') → 'http://...'"""
    if not _GELADEN:
        laden()
    teile = schluessel.split(".")
    wert: Any = _CFG
    for teil in teile:
        if not isinstance(wert, dict):
            return standard
        wert = wert.get(teil, standard)
        if wert is None and teil != teile[-1]:
            return standard
    return wert


def alle() -> dict[str, Any]:
    if not _GELADEN:
        laden()
    return copy.deepcopy(_CFG)
