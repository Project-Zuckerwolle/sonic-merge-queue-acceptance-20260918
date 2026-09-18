"""Nova Predator v3.4 — Context-Cleanup.

Löscht automatisch alte Upload-Dateien aus workspace_agent/context/.
Wird vom Scheduler regelmäßig aufgerufen.

Standard-TTL: 24 Stunden (konfigurierbar über config.yaml)
  context_cleanup.ttl_stunden: 24

Warum TTL statt manuellem Löschen:
  - User uploadt Dateien als Kontext für ein Gespräch
  - Nach dem Gespräch sind sie nicht mehr nötig
  - Nova läuft lokal → kein externer Storage-Druck, aber Ordnung muss sein
  - workspace_agent/context/ soll kein Friedhof werden
"""
from __future__ import annotations

import time
from pathlib import Path

from core.logger import get

log = get("context_cleanup")

CONTEXT_PFAD   = Path("workspace_agent/context")
DEFAULT_TTL_H  = 24   # Stunden


async def ausfuehren(ttl_stunden: float = DEFAULT_TTL_H) -> None:
    """Löscht Dateien in workspace_agent/context/ die älter als ttl_stunden sind.

    Löscht auch leere Unterordner die nach dem Cleanup übrig bleiben.
    Berührt workspace_agent/ selbst und seine anderen Unterordner nicht.
    """
    if not CONTEXT_PFAD.exists():
        return

    jetzt      = time.time()
    ttl_sek    = ttl_stunden * 3600
    geloescht  = 0
    fehler     = 0

    # Alle Dateien rekursiv durchgehen
    for datei in list(CONTEXT_PFAD.rglob("*")):
        if not datei.is_file():
            continue
        try:
            alter_sek = jetzt - datei.stat().st_mtime
            if alter_sek > ttl_sek:
                datei.unlink()
                geloescht += 1
                log.debug("Gelöscht (%.1fh alt): %s", alter_sek / 3600, datei.name)
        except Exception as e:
            log.warning("Cleanup-Fehler bei %s: %s", datei, e)
            fehler += 1

    # Leere Unterordner entfernen (von unten nach oben)
    for ordner in sorted(CONTEXT_PFAD.rglob("*"), reverse=True):
        if ordner.is_dir() and ordner != CONTEXT_PFAD:
            try:
                ordner.rmdir()   # schlägt fehl wenn nicht leer → ignorieren
                log.debug("Leerer Ordner entfernt: %s", ordner.name)
            except OSError:
                pass   # nicht leer — ok

    if geloescht > 0 or fehler > 0:
        log.info(
            "Context-Cleanup: %d Datei(en) gelöscht, %d Fehler "
            "(TTL=%.0fh, Pfad=%s)",
            geloescht, fehler, ttl_stunden, CONTEXT_PFAD,
        )
