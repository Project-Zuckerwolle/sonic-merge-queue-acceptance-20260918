"""Nova Predator v2 — ThermalMonitor.
Überwacht GPU/CPU-Temperaturen und löst Alarm-Aktionen aus.

Unterstützte Methoden (Fallback-Chain):
  1. LibreHardwareMonitor WMI  (bestes für AMD auf Windows — braucht LHM installiert)
  2. PowerShell + WMI ADL      (direkt, kein extra Tool nötig)
  3. AIDA64 shared memory      (wenn AIDA64 läuft)
  4. Ollama /api/ps VRAM       (immer verfügbar — zeigt VRAM-Last)

Alarmgrenzen (konfigurierbar in config.yaml):
  thermal.gpu_warn_c:    80   # GPU-Warnung (Orange-Badge in UI)
  thermal.gpu_crit_c:    95   # GPU-Kritisch → LLM-Drossel + Toast
  thermal.cpu_warn_c:    85
  thermal.cpu_crit_c:    95

Bei kritischer Temperatur:
  → Alle LLMs sofort entladen (VRAM frei, GPU macht weniger)
  → EventBus: THERMAL_ALARM
  → WS-Event an alle Clients: thermal_alarm

Python 3.14 kompatibel: asyncio.to_thread() für subprocess-Aufrufe.
"""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
_NO_WINDOW = 0x08000000 if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.event_bus import EventBus, EventTyp
from core.logger import get

if TYPE_CHECKING:
    from core.ollama_client import OllamaClient

log = get("thermal_monitor")

# Neue EventTypen — werden zur EventBus-Enum hinzugefügt wenn noch nicht vorhanden
_THERMAL_WARN  = "thermal.warn"
_THERMAL_ALARM = "thermal.alarm"
_THERMAL_OK    = "thermal.ok"


@dataclass
class ThermalReading:
    """Aktuelle Temperatur-Messung."""
    gpu_temp_c:   float | None = None   # GPU Core-Temperatur
    gpu_hotspot_c: float | None = None  # GPU Hotspot (falls verfügbar)
    cpu_temp_c:   float | None = None
    vram_gb:      float = 0.0           # VRAM-Auslastung
    gpu_load_pct: float | None = None   # GPU-Auslastung %
    quelle:       str = "unknown"       # Woher die Werte kommen
    zeitpunkt:    float = field(default_factory=time.monotonic)


@dataclass
class ThermalGrenzen:
    gpu_warn_c:  float = 80.0
    gpu_crit_c:  float = 95.0
    cpu_warn_c:  float = 85.0
    cpu_crit_c:  float = 95.0


class ThermalMonitor:
    """Hintergrundservice für Temperatur-Überwachung.

    Startet mit .starten(), stoppt mit .stoppen().
    Letzte Messung via .letztes_reading.
    """

    def __init__(
        self,
        bus: EventBus,
        ollama: "OllamaClient | None" = None,
        grenzen: ThermalGrenzen | None = None,
        poll_intervall_s: int = 10,
    ) -> None:
        self._bus              = bus
        self._ollama           = ollama
        self._grenzen          = grenzen or ThermalGrenzen()
        self._poll_s           = poll_intervall_s
        self._letztes: ThermalReading = ThermalReading()
        self._laeuft           = False
        self._task: asyncio.Task | None = None
        self._alarm_aktiv      = False   # Verhindert Spam
        self._lhm_verfuegbar   = None    # Lazy check

    # ── Public API ───────────────────────────────────────────────────

    async def starten(self) -> None:
        if self._laeuft:
            return
        self._laeuft = True
        self._task = asyncio.create_task(self._zyklus(), name="thermal_monitor")
        log.info("ThermalMonitor gestartet (Intervall: %ds)", self._poll_s)

    async def stoppen(self) -> None:
        self._laeuft = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    @property
    def letztes_reading(self) -> ThermalReading:
        return self._letztes

    def ist_kritisch(self) -> bool:
        r = self._letztes
        if r.gpu_temp_c and r.gpu_temp_c >= self._grenzen.gpu_crit_c:
            return True
        if r.cpu_temp_c and r.cpu_temp_c >= self._grenzen.cpu_crit_c:
            return True
        return False

    def ist_warm(self) -> bool:
        """True wenn Temperatur im Warn-Bereich ODER kritisch.
        Beide Flags können gleichzeitig True sein.
        """ 
        r = self._letztes
        if r.gpu_temp_c and r.gpu_temp_c >= self._grenzen.gpu_warn_c:
            return True
        if r.cpu_temp_c and r.cpu_temp_c >= self._grenzen.cpu_warn_c:
            return True
        return False

    def als_dict(self) -> dict:
        r = self._letztes
        return {
            "gpu_temp_c":    r.gpu_temp_c,
            "gpu_hotspot_c": r.gpu_hotspot_c,
            "cpu_temp_c":    r.cpu_temp_c,
            "vram_gb":       r.vram_gb,
            "gpu_load_pct":  r.gpu_load_pct,
            "quelle":        r.quelle,
            "kritisch":      self.ist_kritisch(),
            "warm":          self.ist_warm(),
        }

    def update_grenzen(self, gpu_warn: float = 80, gpu_crit: float = 95,
                       cpu_warn: float = 85, cpu_crit: float = 95) -> None:
        self._grenzen = ThermalGrenzen(
            gpu_warn_c=gpu_warn, gpu_crit_c=gpu_crit,
            cpu_warn_c=cpu_warn, cpu_crit_c=cpu_crit,
        )

    # ── Mess-Zyklus ──────────────────────────────────────────────────

    async def _zyklus(self) -> None:
        while self._laeuft:
            try:
                reading = await self._messen()
                self._letztes = reading
                await self._bewerten(reading)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.debug("Thermal-Poll Fehler: %s", e)
            await asyncio.sleep(self._poll_s)

    async def _messen(self) -> ThermalReading:
        """Versucht Temperaturen zu lesen — Fallback-Chain."""

        # Methode 1: LibreHardwareMonitor WMI (beste Qualität)
        if self._lhm_verfuegbar is None:
            self._lhm_verfuegbar = await asyncio.to_thread(self._check_lhm)

        if self._lhm_verfuegbar:
            r = await asyncio.to_thread(self._lese_lhm)
            if r:
                return r

        # Methode 2: PowerShell direkt
        r = await asyncio.to_thread(self._lese_powershell)
        if r:
            return r

        # Methode 3: Ollama VRAM (immer verfügbar wenn Ollama läuft)
        if self._ollama:
            try:
                modelle = await self._ollama.geladene_modelle()
                vram = sum(m.get("vram_gb", 0) for m in modelle)
                return ThermalReading(vram_gb=vram, quelle="ollama")
            except Exception:
                pass

        return ThermalReading(quelle="unavailable")

    def _check_lhm(self) -> bool:
        """Prüft ob LibreHardwareMonitor WMI-Namespace verfügbar ist."""
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-WmiObject -Namespace root/LibreHardwareMonitor -Class Sensor -ErrorAction Stop | Select-Object -First 1 Name | Out-String"],
                capture_output=True, encoding="utf-8", text=True, timeout=3, creationflags=_NO_WINDOW)
            return result.returncode == 0 and "Name" in result.stdout
        except Exception:
            return False

    def _lese_lhm(self) -> ThermalReading | None:
        """Liest Temperaturen aus LibreHardwareMonitor WMI."""
        try:
            ps_script = """
$sensors = Get-WmiObject -Namespace root/LibreHardwareMonitor -Class Sensor -ErrorAction Stop
$result = @{}
foreach ($s in $sensors) {
    if ($s.SensorType -eq 'Temperature') {
        $result[$s.Name] = $s.Value
    }
    if ($s.SensorType -eq 'Load' -and $s.Name -like '*GPU Core*') {
        $result['GPU Load'] = $s.Value
    }
}
$result | ConvertTo-Json
"""
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True, encoding="utf-8", text=True, timeout=5, creationflags=_NO_WINDOW)
            if result.returncode != 0 or not result.stdout.strip():
                return None

            data = json.loads(result.stdout)
            gpu_temp  = None
            gpu_hot   = None
            cpu_temp  = None
            gpu_load  = data.get("GPU Load")

            for k, v in data.items():
                k_low = k.lower()
                if "gpu core" in k_low and "temperature" not in k_low:
                    gpu_temp = float(v) if v else gpu_temp
                elif "hotspot" in k_low or "junction" in k_low:
                    gpu_hot = float(v) if v else gpu_hot
                elif "cpu package" in k_low or "cpu temp" in k_low:
                    cpu_temp = float(v) if v else cpu_temp

            if gpu_temp is None and gpu_hot is None and cpu_temp is None:
                return None

            return ThermalReading(
                gpu_temp_c=gpu_temp,
                gpu_hotspot_c=gpu_hot,
                cpu_temp_c=cpu_temp,
                gpu_load_pct=float(gpu_load) if gpu_load else None,
                quelle="LibreHardwareMonitor",
            )
        except Exception as e:
            log.debug("LHM-Lesen Fehler: %s", e)
            return None

    def _lese_powershell(self) -> ThermalReading | None:
        """Liest GPU-Temp via PowerShell + WMI (ohne extra Tools)."""
        try:
            # AMD-spezifisch: Radeon-Adapter-Daten via WMI
            ps_script = r"""
$gpu = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like '*Radeon*' -or $_.Name -like '*AMD*' } | Select-Object -First 1
if ($gpu) {
    Write-Output "GPU_NAME:$($gpu.Name)"
    Write-Output "GPU_DRIVER:$($gpu.DriverVersion)"
}
# CPU Temp via MSAcpi
try {
    $cpu = Get-WmiObject -Namespace root/WMI -Class MSAcpi_ThermalZoneTemperature -ErrorAction Stop
    $tempC = ($cpu[0].CurrentTemperature - 2732) / 10
    Write-Output "CPU_TEMP:$tempC"
} catch {}
"""
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True, encoding="utf-8", text=True, timeout=5, creationflags=_NO_WINDOW)
            if result.returncode != 0:
                return None

            out = result.stdout
            cpu_temp = None
            m = re.search(r"CPU_TEMP:([\d.]+)", out)
            if m:
                cpu_temp = float(m.group(1))

            # Wenn GPU-Name gefunden → zumindest CPU-Temp zurückgeben
            if "GPU_NAME:" in out or cpu_temp:
                return ThermalReading(
                    cpu_temp_c=cpu_temp,
                    quelle="PowerShell/WMI",
                )
        except Exception as e:
            log.debug("PowerShell Thermal Fehler: %s", e)
        return None

    # ── Alarm-Logik ──────────────────────────────────────────────────

    async def _bewerten(self, r: ThermalReading) -> None:
        """Prüft ob Alarm ausgelöst werden soll."""
        if self.ist_kritisch():
            if not self._alarm_aktiv:
                self._alarm_aktiv = True
                log.warning(
                    "THERMAL ALARM: GPU=%.0f°C CPU=%.0f°C",
                    r.gpu_temp_c or 0, r.cpu_temp_c or 0,
                )
                await self._bus.publish(_THERMAL_ALARM, self.als_dict())
        elif self.ist_warm():
            if self._alarm_aktiv:
                self._alarm_aktiv = False
                await self._bus.publish(_THERMAL_WARN, self.als_dict())
        else:
            if self._alarm_aktiv:
                self._alarm_aktiv = False
                await self._bus.publish(_THERMAL_OK, self.als_dict())
