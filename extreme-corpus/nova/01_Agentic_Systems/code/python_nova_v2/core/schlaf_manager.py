"""Nova Predator v3.3 — SchlafManager.
Verwaltet den Sleep-Modus: entlädt Chat-LLM nach Inaktivität,
behält Brain-LLM im VRAM. Weckt Nova bei neuer Nachricht.

v3.3: PC-Standby-Schutz via SetThreadExecutionState direkt im Python-Prozess
(nicht in Subprocess). Verhindert dass Windows den PC in Ruhezustand versetzt
solange Nova läuft.
"""
from __future__ import annotations
import asyncio
import ctypes
import os
import subprocess
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

from core.event_bus import EventBus, EventTyp
from core.logger import get
from core.task_manager import TaskManager

if TYPE_CHECKING:
    from core.ollama_client import OllamaClient

log = get("schlaf_manager")

_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# Windows SetThreadExecutionState Flags
# ES_CONTINUOUS = 0x80000000 — Zustand bleibt bis nächster Aufruf
# ES_SYSTEM_REQUIRED = 0x00000001 — verhindert System-Schlaf
# ES_DISPLAY_REQUIRED = 0x00000002 — verhindert Monitor-Schlaf
_ES_CONTINUOUS       = 0x80000000
_ES_SYSTEM_REQUIRED  = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002
_ES_AWAYMODE         = 0x00000040   # "Away Mode" — PC bleibt für Hintergrundaufgaben aktiv


def _set_execution_state(flags: int) -> None:
    """Setzt Windows SetThreadExecutionState im aktuellen Python-Prozess.
    Nur auf Windows — auf anderen Systemen no-op.
    """
    if os.name != "nt":
        return
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(flags)  # type: ignore[attr-defined]
    except Exception as e:
        log.debug("SetThreadExecutionState fehlgeschlagen: %s", e)


class NovaZustand(str, Enum):
    WACH = "wach"
    SCHLAEFT = "schlaeft"
    AUFWACHEN = "aufwachen"


class SchlafManager:
    """
    Sleep-Modus Logik:
    - Nova läuft: PC-Schlaf permanent verhindert (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    - Nach timeout_s Inaktivität: Chat-LLM entladen (Software-Schlaf)
    - PC geht NIE automatisch in Standby/Ruhezustand solange Nova läuft
    - standby_timeout_s > 0: nach N Stunden Inaktivität PC bewusst in Standby
      versetzen (opt-in, default 0 = nie)
    """

    def __init__(
        self,
        ollama: "OllamaClient",
        task_manager: TaskManager,
        event_bus: EventBus,
        timeout_s: int = 300,
        chat_modell: str = "gemma4:e4b",
        brain_modell: str = "qwen2.5:3b",
        aktiv: bool = True,
        standby_timeout_s: int = 0,   # 0 = PC geht NIE automatisch in Standby
    ) -> None:
        self._ollama             = ollama
        self._tm                 = task_manager
        self._bus                = event_bus
        self._timeout_s          = timeout_s
        self._standby_timeout_s  = standby_timeout_s
        self._chat_modell        = chat_modell
        self._brain_modell       = brain_modell
        self._aktiv              = aktiv
        self._zustand            = NovaZustand.WACH
        self._letzte_aktivitaet: float = 0.0
        self._wacht_task: asyncio.Task | None = None

    async def starten(self) -> None:
        """Startet den Inaktivitäts-Wächter und verhindert PC-Schlaf."""
        if not self._aktiv:
            return

        # PC-Schlaf permanent verhindern solange Nova läuft
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED = 0x80000001
        # Das Flag bleibt aktiv bis Nova beendet wird oder es explizit aufgehoben wird
        _set_execution_state(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
        log.info("PC-Schlaf verhindert (SetThreadExecutionState im Python-Prozess)")

        self._letzte_aktivitaet = self._jetzt()
        self._wacht_task = asyncio.create_task(
            self._inaktivitaets_wacht(), name="schlaf_wacht"
        )
        standby_info = (
            f", opt-in Standby nach {self._standby_timeout_s // 3600}h"
            if self._standby_timeout_s > 0 else " (kein auto-Standby)"
        )
        log.info("SchlafManager aktiv (LLM-Schlaf=%ds%s)", self._timeout_s, standby_info)

    async def stoppen(self) -> None:
        """Stoppt den Wächter und gibt PC-Schlaf wieder frei."""
        if self._wacht_task and not self._wacht_task.done():
            self._wacht_task.cancel()
            try:
                await self._wacht_task
            except asyncio.CancelledError:
                pass
        # Beim Beenden: Execution-State zurücksetzen (PC kann wieder schlafen)
        _set_execution_state(_ES_CONTINUOUS)
        log.info("SetThreadExecutionState zurückgesetzt")

    def aktivitaet_melden(self) -> None:
        """Jede Nutzer-Interaktion ruft dies auf — setzt Timer zurück."""
        self._letzte_aktivitaet = self._jetzt()
        if self._zustand == NovaZustand.SCHLAEFT:
            asyncio.create_task(self._aufwachen(), name="schlaf_aufwachen")

    async def warten_bis_wach(self) -> None:
        """Blockiert bis Nova wach ist. Für WebSocket-Handler vor LLM-Call."""
        while self._zustand != NovaZustand.WACH:
            await asyncio.sleep(0.2)

    @property
    def zustand(self) -> NovaZustand:
        return self._zustand

    @property
    def schlaeft(self) -> bool:
        return self._zustand == NovaZustand.SCHLAEFT

    async def _inaktivitaets_wacht(self) -> None:
        while True:
            await asyncio.sleep(30)
            vergangen = self._jetzt() - self._letzte_aktivitaet

            # LLM-Schlaf
            if self._zustand == NovaZustand.WACH and vergangen >= self._timeout_s:
                await self._einschlafen()

            # Opt-in PC-Standby (nur wenn explizit konfiguriert)
            if (
                self._standby_timeout_s > 0
                and vergangen >= self._standby_timeout_s
                and self._zustand == NovaZustand.SCHLAEFT
            ):
                await self._pc_standby()

    async def _einschlafen(self) -> None:
        if self._zustand != NovaZustand.WACH:
            return
        log.info("Nova schläft ein — Chat-LLM wird entladen")
        self._zustand = NovaZustand.SCHLAEFT
        await self._bus.publish(EventTyp.SCHLAF_BEGINN, {"modell": self._chat_modell})
        ok = await self._ollama.modell_entladen(self._chat_modell)
        if ok:
            log.info("Chat-LLM '%s' entladen — VRAM frei", self._chat_modell)
        else:
            log.warning("Chat-LLM entladen fehlgeschlagen")

    async def _aufwachen(self) -> None:
        if self._zustand != NovaZustand.SCHLAEFT:
            return
        self._zustand = NovaZustand.AUFWACHEN
        log.info("Nova wacht auf — Chat-LLM wird geladen...")
        await self._bus.publish(EventTyp.SCHLAF_ENDE, {"modell": self._chat_modell})
        ok = await self._ollama.modell_laden(self._chat_modell)
        if ok:
            log.info("Chat-LLM bereit")
        else:
            log.warning("Chat-LLM laden fehlgeschlagen")
        self._zustand = NovaZustand.WACH
        self._letzte_aktivitaet = self._jetzt()

    async def _pc_standby(self) -> None:
        """Opt-in PC-Standby nach standby_timeout_s Stunden.
        Nur wenn standby_timeout_s > 0 in config.yaml gesetzt ist.
        Bevor Standby: Execution-State zurücksetzen damit Windows Standby erlaubt.
        """
        log.info("Opt-in PC-Standby nach %dh Inaktivität",
                 self._standby_timeout_s // 3600)
        await self._bus.publish("schlaf.standby", {
            "zeitpunkt": datetime.now(timezone.utc).isoformat(),
        })
        # Execution-State zurücksetzen damit SetSuspendState wirken kann
        _set_execution_state(_ES_CONTINUOUS)
        try:
            if os.name == "nt":
                ps_cmd = (
                    "Add-Type -TypeDefinition '"
                    "using System; using System.Runtime.InteropServices; "
                    "public class PwrMgmt { "
                    "[DllImport(\"Powrprof.dll\", SetLastError=true)] "
                    "public static extern bool SetSuspendState("
                    "bool hibernate, bool forceCritical, bool disableWakeEvent); "
                    "}'; "
                    "[PwrMgmt]::SetSuspendState($false, $false, $false) | Out-Null"
                )
                await asyncio.to_thread(
                    subprocess.run,
                    ["powershell", "-NoProfile", "-Command", ps_cmd],
                    capture_output=True, encoding="utf-8", timeout=10,
                    creationflags=_NO_WINDOW,
                )
            else:
                await asyncio.to_thread(
                    subprocess.run, ["systemctl", "suspend"],
                    capture_output=True, timeout=10,
                )
            log.info("PC-Standby ausgelöst")
        except Exception as e:
            log.warning("PC-Standby fehlgeschlagen: %s", e)
        finally:
            # Nach Wakeup: Execution-State wieder setzen
            _set_execution_state(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)

    def _jetzt(self) -> float:
        import time
        return time.monotonic()

    def status(self) -> dict:
        return {
            "zustand":           self._zustand.value,
            "timeout_s":         self._timeout_s,
            "standby_timeout_s": self._standby_timeout_s,
            "aktiv":             self._aktiv,
            "chat_modell":       self._chat_modell,
        }

