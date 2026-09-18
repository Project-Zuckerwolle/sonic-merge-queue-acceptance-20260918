"""Nova Predator v1 Layer 6 — Computer Controller.

Steuert Windows-Apps und System-Funktionen.
Nur auf Windows verfügbar. Auf anderen Systemen: Fehlermeldung.

Abhängigkeiten (werden lazy geprüft):
  - psutil       — Prozesse, Speicher (immer verfügbar)
  - pygetwindow  — Fenster-Management (optional)
  - pycaw        — Windows Audio (optional, Windows-only)
  - pyperclip    — Clipboard (optional)
"""
from __future__ import annotations

import os
import platform
import subprocess
_NO_WINDOW = 0x08000000 if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
import sys
from typing import Any

from core.logger import get

log = get("apex.computer")

# Bekannte App-Namen → ausführbare Pfade (Windows)
APP_WHITELIST: dict[str, list[str]] = {
    "chrome":       ["chrome.exe", "Google Chrome"],
    "firefox":      ["firefox.exe"],
    "notepad":      ["notepad.exe"],
    "notepad++":    ["notepad++.exe"],
    "explorer":     ["explorer.exe"],
    "calculator":   ["calc.exe"],
    "spotify":      ["Spotify.exe"],
    "vscode":       ["Code.exe", "code.exe"],
    "terminal":     ["wt.exe", "cmd.exe"],
    "cmd":          ["cmd.exe"],
    "powershell":   ["powershell.exe"],
    "taskmgr":      ["taskmgr.exe"],
    "word":         ["WINWORD.EXE"],
    "excel":        ["EXCEL.EXE"],
    "outlook":      ["OUTLOOK.EXE"],
    "paint":        ["mspaint.exe"],
    "vlc":          ["vlc.exe"],
    "discord":      ["Discord.exe"],
    "teams":        ["Teams.exe"],
    "zoom":         ["Zoom.exe"],
}


class ComputerController:
    """Steuert Windows-Apps und System-Funktionen.

    Alle Methoden sind sync (werden im run_in_executor aufgerufen)
    oder async wenn Netzwerk-Calls nötig.
    """

    def __init__(self) -> None:
        self._ist_windows = platform.system() == "Windows"
        if not self._ist_windows:
            log.info("ComputerController: Nicht-Windows — App-Steuerung eingeschränkt")

    # ── App-Verwaltung ────────────────────────────────────────────────────────

    def app_starten(self, args: dict) -> str:
        """Startet eine Anwendung per Namen oder Pfad."""
        app_name = args.get("app", args.get("name", "")).lower().strip()

        if not app_name:
            return "Fehler: Kein App-Name angegeben"

        # Whitelist prüfen
        kandidaten = APP_WHITELIST.get(app_name, [app_name])

        if self._ist_windows:
            for kandidat in kandidaten:
                try:
                    subprocess.Popen(
                        kandidat,
                        shell=True,
                        creationflags=subprocess.DETACHED_PROCESS,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    log.info("App gestartet: %s", kandidat)
                    return f"✓ {app_name} gestartet"
                except Exception:
                    continue
            return f"✗ {app_name} konnte nicht gestartet werden"
        else:
            # Linux/Mac: Bestmöglich versuchen
            try:
                subprocess.Popen(
                    kandidaten[0], shell=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                return f"✓ {app_name} gestartet"
            except Exception as e:
                return f"✗ Fehler: {e}"

    def app_beenden(self, args: dict) -> str:
        """Beendet eine Anwendung per Namen."""
        import psutil
        app_name = args.get("app", args.get("name", "")).lower()
        if not app_name:
            return "Fehler: Kein App-Name angegeben"

        # Mögliche Prozess-Namen
        kandidaten = APP_WHITELIST.get(app_name, [app_name])
        beendet = 0

        for proc in psutil.process_iter(["name", "pid"]):
            proc_name = (proc.info.get("name") or "").lower()
            for kand in kandidaten:
                if kand.lower().replace(".exe", "") in proc_name or proc_name in kand.lower():
                    try:
                        proc.terminate()
                        beendet += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    break

        if beendet:
            return f"✓ {beendet} Prozess(e) von '{app_name}' beendet"
        return f"✗ Kein Prozess '{app_name}' gefunden"

    def prozesse_liste(self, args: dict | None = None) -> str:
        """Gibt Liste laufender Prozesse zurück."""
        import psutil
        filter_name = (args or {}).get("filter", "").lower()

        prozesse = []
        for proc in psutil.process_iter(["name", "pid", "memory_info", "status"]):
            try:
                name = proc.info.get("name") or ""
                if filter_name and filter_name not in name.lower():
                    continue
                mem_mb = (proc.info.get("memory_info") or type("x", (), {"rss": 0})()).rss // (1024 * 1024)
                prozesse.append(f"{proc.info['pid']:6d} {name[:30]:<30} {mem_mb:4d}MB")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if not prozesse:
            return f"Keine Prozesse{' mit Filter ' + filter_name if filter_name else ''} gefunden"

        header = f"{'PID':>6} {'Name':<30} {'RAM':>5}"
        return header + "\n" + "\n".join(prozesse[:30])

    # ── Volume ────────────────────────────────────────────────────────────────

    def volume_setzen(self, args: dict) -> str:
        """Setzt System-Volume (0-100)."""
        prozent = int(args.get("prozent", args.get("volume", 50)))
        prozent = max(0, min(100, prozent))

        if not self._ist_windows:
            return f"Volume-Steuerung nur auf Windows verfügbar (gewünscht: {prozent}%)"

        try:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore
            import comtypes
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None)
            volume = interface.QueryInterface(IAudioEndpointVolume)
            # pycaw erwartet Wert zwischen 0.0 und 1.0
            volume.SetMasterVolumeLevelScalar(prozent / 100.0, None)
            return f"✓ Volume auf {prozent}% gesetzt"
        except ImportError:
            # Fallback: nircmd wenn verfügbar
            try:
                subprocess.run(
                    f"nircmd.exe setsysvolume {int(prozent / 100 * 65535)}",
                    shell=True, capture_output=True
                )
                return f"✓ Volume auf {prozent}% gesetzt (via nircmd)"
            except Exception:
                pass
            return "✗ pycaw nicht installiert (pip install pycaw)"
        except Exception as e:
            return f"✗ Volume-Fehler: {e}"

    # ── Screenshot ────────────────────────────────────────────────────────────

    def screenshot(self, args: dict | None = None) -> str:
        """Macht Screenshot und speichert ihn."""
        try:
            import PIL.ImageGrab  # type: ignore
            from datetime import datetime
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            pfad = Path("workspace_apex") / f"screenshot_{ts}.png"
            pfad.parent.mkdir(parents=True, exist_ok=True)
            img  = PIL.ImageGrab.grab()
            img.save(str(pfad))
            return f"✓ Screenshot gespeichert: {pfad.name} ({img.size[0]}x{img.size[1]})"
        except ImportError:
            return "✗ Pillow nicht installiert (pip install Pillow)"
        except Exception as e:
            return f"✗ Screenshot-Fehler: {e}"

    # ── Clipboard ─────────────────────────────────────────────────────────────

    def clipboard_setzen(self, args: dict) -> str:
        """Kopiert Text in Zwischenablage."""
        text = args.get("text", "")
        if not text:
            return "Fehler: Kein Text angegeben"
        try:
            import pyperclip  # type: ignore
            pyperclip.copy(text)
            return f"✓ Text in Clipboard ({len(text)} Zeichen)"
        except ImportError:
            if self._ist_windows:
                try:
                    subprocess.run(
                        f'echo {text[:200]} | clip',
                        shell=True, capture_output=True
                    , creationflags=_NO_WINDOW)
                    return f"✓ Text in Clipboard (via clip)"
                except Exception:
                    pass
            return "✗ pyperclip nicht installiert (pip install pyperclip)"
        except Exception as e:
            return f"✗ Clipboard-Fehler: {e}"

    # ── System-Info ───────────────────────────────────────────────────────────

    def system_info(self, args: dict | None = None) -> str:
        """Gibt System-Informationen zurück."""
        import psutil
        cpu    = psutil.cpu_percent(interval=0.5)
        ram    = psutil.virtual_memory()
        disk   = psutil.disk_usage("/")
        ram_gb = ram.total / (1024**3)
        ram_pct= ram.percent

        lines = [
            f"CPU: {cpu:.0f}%",
            f"RAM: {ram.used//(1024**2)}MB / {ram_gb:.1f}GB ({ram_pct:.0f}%)",
            f"Disk: {disk.used//(1024**3)}GB / {disk.total//(1024**3)}GB ({disk.percent:.0f}%)",
            f"Platform: {platform.system()} {platform.release()}",
        ]
        return "\n".join(lines)
