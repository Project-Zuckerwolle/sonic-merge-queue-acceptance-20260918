"""Nova Predator v2 — PowerSave Manager.
Strom-sparen wenn Nova idle ist — ohne PC-Sleep (Gaming-sicher).

Was passiert bei PowerSave ON:
  1. Alle LLMs aus VRAM entladen (GPU-Lüfter beruhigen sich)
  2. Windows Power-Plan → "Energiesparmodus" (powercfg)
  3. Bildschirm aus (Monitor-Sleep, PC bleibt an + erreichbar)

Was passiert bei PowerSave OFF (Aufwachen):
  1. Power-Plan → "Ausgewogen" (Balanced)
  2. Bildschirm an
  3. Chat-LLM laden (bereit für Gespräch)

Nova bleibt immer erreichbar:
  - FastAPI läuft weiter
  - WebSocket nimmt Verbindungen an
  - Handy kann jederzeit aufwecken

Thermal-Integration:
  - Bei kritischer GPU-Temp → PowerSave aktivieren (Notfall-Kühlung)
  - Danach automatisch wieder aufwecken wenn unter Warn-Grenze

Python 3.14 kompatibel: asyncio.to_thread() für subprocess-Aufrufe.
"""
from __future__ import annotations

import asyncio
import ctypes
import os
import subprocess

# Windows: verhindert dass PowerShell-Fenster aufpoppt
_NO_WINDOW = 0x08000000 if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from core.event_bus import EventBus, EventTyp
from core.logger import get

if TYPE_CHECKING:
    from core.ollama_client import OllamaClient
    from core.schlaf_manager import SchlafManager

log = get("powersave")

# Windows Power-Plan GUIDs
_PLAN_BALANCED    = "381b4222-f694-41f0-9685-ff5bb260df2e"
_PLAN_POWERSAVER  = "a1841308-3541-4fab-bc81-f71556f20b4a"
_PLAN_PERFORMANCE = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"


class PowerSaveManager:
    """Verwaltet den Strom-Spar-Modus für Nova.

    Kein PC-Sleep — nur Monitor aus + LLMs entladen + Stromplan.
    Nova bleibt jederzeit erreichbar.
    """

    def __init__(
        self,
        ollama: "OllamaClient",
        bus: EventBus,
        chat_modell: str = "gemma4:e4b",
        aktiv: bool = False,         # Start: aus (User muss explizit einschalten)
    ) -> None:
        self._ollama       = ollama
        self._bus          = bus
        self._chat_modell  = chat_modell
        self._powersave_an = aktiv
        self._eingeschlafen_um: str | None = None

    # ── Public API ───────────────────────────────────────────────────

    async def einschalten(self) -> None:
        """PowerSave aktivieren — LLMs entladen, Monitor aus, Stromplan."""
        if self._powersave_an:
            return
        log.info("PowerSave: aktiviere Strom-Spar-Modus")
        self._powersave_an = True
        self._eingeschlafen_um = datetime.now(timezone.utc).isoformat()

        # 1. Alle LLMs entladen
        await self._llms_entladen()

        # 2. Windows Power-Plan → Energiesparmodus
        await asyncio.to_thread(self._power_plan_setzen, _PLAN_POWERSAVER)

        # 3. Bildschirm aus (Monitor-Sleep — PC bleibt an)
        await asyncio.to_thread(self._monitor_aus)

        await self._bus.publish("powersave.an", {
            "zeitpunkt": self._eingeschlafen_um,
        })
        log.info("PowerSave aktiv — Nova schläft, ist aber erreichbar")

    async def ausschalten(self) -> None:
        """PowerSave deaktivieren — Monitor an, Power-Plan zurück, LLM laden."""
        if not self._powersave_an:
            return
        log.info("PowerSave: deaktiviere Strom-Spar-Modus")
        self._powersave_an = False

        # 1. Power-Plan → Balanced
        await asyncio.to_thread(self._power_plan_setzen, _PLAN_BALANCED)

        # 2. Monitor an
        await asyncio.to_thread(self._monitor_an)

        # 3. Chat-LLM laden
        await self._ollama.modell_laden(self._chat_modell)

        await self._bus.publish("powersave.aus", {
            "geschlafen_seit": self._eingeschlafen_um,
        })
        self._eingeschlafen_um = None
        log.info("PowerSave deaktiviert — Nova bereit")

    async def toggle(self) -> bool:
        """Umschalten. Gibt neuen Zustand zurück."""
        if self._powersave_an:
            await self.ausschalten()
        else:
            await self.einschalten()
        return self._powersave_an

    async def notfall_kuehlung(self) -> None:
        """Sofort alle LLMs entladen bei Thermal-Alarm (ohne Monitor aus)."""
        log.warning("Notfall-Kühlung: entlade alle LLMs wegen Thermal-Alarm")
        await self._llms_entladen()
        await self._bus.publish("powersave.notfall", {})

    @property
    def ist_aktiv(self) -> bool:
        return self._powersave_an

    def status(self) -> dict:
        return {
            "aktiv": self._powersave_an,
            "eingeschlafen_um": self._eingeschlafen_um,
            "chat_modell": self._chat_modell,
        }

    # ── Windows-Aktionen ─────────────────────────────────────────────

    def _power_plan_setzen(self, plan_guid: str) -> None:
        """Setzt Windows Power-Plan via powercfg."""
        try:
            subprocess.run(
                ["powercfg", "/setactive", plan_guid],
                capture_output=True, encoding="utf-8", timeout=5,
                creationflags=_NO_WINDOW,
            )
            plan_name = {
                _PLAN_POWERSAVER:  "Energiesparmodus",
                _PLAN_BALANCED:    "Ausgewogen",
                _PLAN_PERFORMANCE: "Höchstleistung",
            }.get(plan_guid, plan_guid[:8])
            log.info("Power-Plan: %s", plan_name)
        except Exception as e:
            log.debug("Power-Plan Fehler (nur Windows): %s", e)

    def _monitor_aus(self) -> None:
        """Schaltet den Monitor in Energiesparmodus (PC bleibt an).

        v3.3: SetThreadExecutionState direkt im Python-Prozess via ctypes —
        der frühere Subprocess-Ansatz funktionierte nicht weil das Flag nur
        für den PowerShell-Thread galt, der sofort endete.
        """
        try:
            if os.name == "nt":
                # ES_CONTINUOUS | ES_SYSTEM_REQUIRED — PC bleibt wach
                ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
                    0x80000001
                )
                log.info("PC-Sleep verhindert (SetThreadExecutionState direkt im Prozess)")
        except Exception as e:
            log.debug("SetThreadExecutionState Fehler: %s", e)

        try:
            # Monitor aus via SendMessage
            ps = (
                "Add-Type -TypeDefinition '"
                "using System; using System.Runtime.InteropServices; "
                "public class Mon { "
                "[DllImport(\"user32.dll\")] "
                "public static extern int SendMessage(int h, int m, int w, int l); "
                "}'; "
                "[Mon]::SendMessage(0xFFFF, 0x0112, 0xF170, 2) | Out-Null"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                capture_output=True, encoding="utf-8", timeout=5,
                creationflags=_NO_WINDOW,
            )
            log.info("Monitor: aus")
        except Exception as e:
            log.debug("Monitor-Aus Fehler: %s", e)

    def _monitor_an(self) -> None:
        """Weckt den Monitor auf."""
        try:
            ps = (
                "Add-Type -TypeDefinition '"
                "using System; using System.Runtime.InteropServices; "
                "public class Mon { "
                "[DllImport(\"user32.dll\")] "
                "public static extern int SendMessage(int h, int m, int w, int l); "
                "}'; "
                "[Mon]::SendMessage(0xFFFF, 0x0112, 0xF170, -1) | Out-Null"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                capture_output=True, encoding="utf-8", timeout=5,
                creationflags=_NO_WINDOW,
            )
            log.info("Monitor: an")
        except Exception as e:
            log.debug("Monitor-An Fehler: %s", e)

    # ── Hilfsmethoden ────────────────────────────────────────────────

    async def _llms_entladen(self) -> None:
        """Entlädt alle aktuell geladenen LLMs."""
        try:
            modelle = await self._ollama.geladene_modelle()
            for m in modelle:
                await self._ollama.modell_entladen(m["name"])
                log.debug("Entladen: %s", m["name"])
        except Exception as e:
            log.debug("LLM-Entladen Fehler: %s", e)
