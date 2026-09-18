"""Nova v9 – Konfiguration.

Laedt config.yaml, merged mit Defaults, bietet type-safe get().
Unveraendert von v8 uebernommen, Version auf 9.0 aktualisiert.
"""
from __future__ import annotations
import copy
from pathlib import Path
from typing import Any

import yaml

from core.logger import get

log = get("config")

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULTS: dict[str, Any] = {
    "nova": {"name": "Nova", "version": "9.0"},
    "ollama": {
        "host":        "http://localhost:11434",
        "modell":      "gemma4:e4b",
        "temperature": 1.0,
        "top_p":       0.95,
        "top_k":       64,
        "max_tokens":  2048,
    },
    "modelle": {
        "chat":    "gemma4:e4b",
        "schnell": "gemma4:e4b",
        "code":    "qwen2.5-coder:7b",
        "embed":   "nomic-embed-text",
    },
    "brain": {
        "pfad":                    "./brain",
        "connections_datei":       "./data/_connections.json",
        "tags_datei":              "./data/_tags.json",
        "fuzzy_duplikat_schwelle": 85,
    },
    "memory": {
        "session_pfad":             "./memory/sessions",
        "archiv_pfad":              "./memory/archives",
        "session_dauer_stunden":    24,
        "max_kontext_nachrichten":  8,
        "embed_datei":              "./data/memory_embeddings.npy",
        "embed_ids_datei":          "./data/memory_embed_ids.json",
    },
    "skills_pfad": "./skills",
    "router": {
        "min_score":   0.72,   # Mindest-Cosinus-Score (v8 hatte 0.60-0.68, zu niedrig)
        "max_skills":  2,      # Maximal 2 Skills gleichzeitig (v8 feuerte alle 10)
    },
    "context": {
        "max_tokens":        6000,
        "compress_schwelle": 0.85,
        "compress_ziel":     0.50,
    },
    "thinker": {
        "entries_schwelle": 5,
    },
    "scheduler": {
        "briefing_zeit":      "07:00",
        "thinker_interval_h": 6,
    },
    "workspace": {"pfad": "./workspace"},
    "sandbox":   {"timeout": 30},
    "web":       {"host": "127.0.0.1", "port": 8000},
    "voice": {
        "aktiviert":  False,
        "stt_modell": "base",
        "tts_stimme": "de_DE-thorsten-medium",
    },
    "logging": {
        # debug_modus: true  -> alles auf Bildschirm + in nova.log
        # debug_modus: false -> INFO auf Bildschirm, DEBUG immer in nova.log
        "debug_modus": False,
    },
}


# ─── Laden + Mergen ───────────────────────────────────────────────────────────

def _deep_merge(basis: dict, override: dict) -> dict:
    """Merged override rekursiv in basis."""
    result = copy.deepcopy(basis)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def laden(pfad: str = "config.yaml") -> dict[str, Any]:
    """Laedt config.yaml und merged mit Defaults."""
    cfg = copy.deepcopy(DEFAULTS)
    p = Path(pfad)
    if p.exists():
        try:
            user = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            cfg = _deep_merge(cfg, user)
            log.info(f"Config geladen: {pfad}")
        except Exception as e:
            log.error(f"Config-Fehler ({pfad}): {e} - nutze Defaults")
    else:
        log.warning("Keine config.yaml gefunden - nutze Defaults")
    return cfg


def get(cfg: dict, *schluessel: str, default: Any = None) -> Any:
    """Sicherer Tiefenzugriff: get(cfg, 'ollama', 'host', default='...')"""
    node = cfg
    for k in schluessel:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node
