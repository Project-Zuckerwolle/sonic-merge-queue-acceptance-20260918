"""Nova Predator v1 Layer 6 — Voice I/O.

STT: faster-whisper (lokal, ROCm-kompatibel auf RX 7900 XT)
TTS: edge-tts (Microsoft, kostenlos, gut) → Fallback pyttsx3 (offline SAPI)

Alle Imports lazy — kein Crash wenn Pakete nicht installiert.
"""
from __future__ import annotations

import asyncio
import io
import platform
import tempfile
from pathlib import Path
from typing import Callable

from core.logger import get

log = get("apex.voice")


class VoiceIO:
    """Sprach-Ein/Ausgabe für Nova APEX.

    STT: faster-whisper → numpy-Array → Text
    TTS: edge-tts → MP3/WAV → pygame abspielen
         Fallback: pyttsx3 (Windows SAPI, kein Internet)
    """

    # TTS-Stimme (edge-tts)
    EDGE_STIMME = "de-DE-KatjaNeural"     # Deutsch, natürlich
    WHISPER_MODELL = "base"               # tiny | base | small | medium

    def __init__(self) -> None:
        self._whisper_modell = None       # Lazy init
        self._tts_engine     = None       # pyttsx3 Fallback
        self._edge_verfuegbar = False
        self._whisper_verfuegbar = False
        self._ist_windows = platform.system() == "Windows"

        # Verfügbarkeit prüfen
        self._pruefen()

    def _pruefen(self) -> None:
        """Prüft welche Voice-Pakete verfügbar sind."""
        try:
            import edge_tts  # noqa
            self._edge_verfuegbar = True
            log.debug("edge-tts verfügbar")
        except ImportError:
            log.debug("edge-tts nicht installiert")

        try:
            import faster_whisper  # noqa
            self._whisper_verfuegbar = True
            log.debug("faster-whisper verfügbar")
        except ImportError:
            log.debug("faster-whisper nicht installiert")

    # ── TTS ──────────────────────────────────────────────────────────────────

    async def spreche(self, text: str, stimme: str | None = None) -> bool:
        """Spricht Text aus. Gibt True zurück wenn erfolgreich."""
        if not text.strip():
            return False

        # edge-tts bevorzugen (bessere Qualität)
        if self._edge_verfuegbar:
            ok = await self._spreche_edge(text, stimme or self.EDGE_STIMME)
            if ok:
                return True

        # Fallback: pyttsx3
        if self._ist_windows:
            return await self._spreche_pyttsx3(text)

        log.warning("Kein TTS-Backend verfügbar")
        return False

    async def _spreche_edge(self, text: str, stimme: str) -> bool:
        """TTS via edge-tts → MP3 → abspielen."""
        try:
            import edge_tts
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_pfad = f.name

            communicate = edge_tts.Communicate(text, stimme)
            await communicate.save(tmp_pfad)

            # Abspielen
            await self._abspielen(tmp_pfad)
            Path(tmp_pfad).unlink(missing_ok=True)
            return True
        except Exception as e:
            log.warning("edge-tts Fehler: %s", e)
            return False

    async def _spreche_pyttsx3(self, text: str) -> bool:
        """TTS via pyttsx3 (Windows SAPI, 100% offline)."""
        try:
            import pyttsx3
            loop = asyncio.get_running_loop()

            def _sync():
                engine = pyttsx3.init()
                engine.setProperty("rate", 175)    # Sprechgeschwindigkeit
                engine.setProperty("volume", 0.9)
                # Deutsche Stimme wählen falls verfügbar
                voices = engine.getProperty("voices")
                for v in voices:
                    if "german" in v.name.lower() or "de" in v.id.lower():
                        engine.setProperty("voice", v.id)
                        break
                engine.say(text)
                engine.runAndWait()

            await loop.run_in_executor(None, _sync)
            return True
        except Exception as e:
            log.warning("pyttsx3 Fehler: %s", e)
            return False

    async def _abspielen(self, pfad: str) -> None:
        """Spielt Audio-Datei ab."""
        try:
            import pygame  # type: ignore
            loop = asyncio.get_running_loop()

            def _sync():
                pygame.mixer.init()
                pygame.mixer.music.load(pfad)
                pygame.mixer.music.play()
                # Warten bis fertig
                while pygame.mixer.music.get_busy():
                    import time
                    time.sleep(0.1)
                pygame.mixer.quit()

            await loop.run_in_executor(None, _sync)
        except ImportError:
            # Fallback: Windows Media Player / afplay
            import subprocess
            if self._ist_windows:
                subprocess.Popen(
                    ["powershell", "-c", f"(New-Object Media.SoundPlayer '{pfad}').PlaySync()"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            else:
                subprocess.Popen(
                    ["afplay", pfad],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=_NO_WINDOW)
            await asyncio.sleep(1)  # Grobe Wartezeit

    # ── STT ──────────────────────────────────────────────────────────────────

    async def hoere(
        self,
        sekunden: float = 5.0,
        callback: Callable[[str], None] | None = None,
    ) -> str | None:
        """Nimmt Mikrofon-Audio auf und transkribiert es.

        callback: wird mit Text aufgerufen sobald Transkription fertig.
        """
        if not self._whisper_verfuegbar:
            log.warning("faster-whisper nicht installiert")
            return None

        try:
            import numpy as np

            # Audio aufnehmen
            audio = await self._aufnehmen(sekunden)
            if audio is None:
                return None

            # Transkribieren
            text = await self._transkribieren(audio)
            if text and callback:
                callback(text)
            return text
        except Exception as e:
            log.error("STT Fehler: %s", e)
            return None

    async def _aufnehmen(self, sekunden: float) -> "np.ndarray | None":
        """Nimmt Mikrofon-Audio auf."""
        try:
            import sounddevice as sd  # type: ignore
            import numpy as np

            loop = asyncio.get_running_loop()

            def _sync():
                log.debug("Aufnahme startet (%.1fs)", sekunden)
                audio = sd.rec(
                    int(sekunden * 16000),
                    samplerate=16000,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()
                return audio.flatten()

            return await loop.run_in_executor(None, _sync)
        except ImportError:
            log.warning("sounddevice nicht installiert (pip install sounddevice)")
            return None
        except Exception as e:
            log.error("Aufnahme-Fehler: %s", e)
            return None

    async def _transkribieren(self, audio: "np.ndarray") -> str | None:
        """Transkribiert Audio mit faster-whisper."""
        try:
            from faster_whisper import WhisperModel  # type: ignore
            import numpy as np

            # Modell lazy laden
            if self._whisper_modell is None:
                loop = asyncio.get_running_loop()
                def _laden():
                    # ROCm/CUDA-Erkennung
                    try:
                        import torch
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        device = "cpu"
                    log.info("Lade Whisper-Modell '%s' auf %s", self.WHISPER_MODELL, device)
                    return WhisperModel(
                        self.WHISPER_MODELL,
                        device=device,
                        compute_type="float32",
                    )
                self._whisper_modell = await loop.run_in_executor(None, _laden)

            loop = asyncio.get_running_loop()

            def _transkribiere():
                segmente, info = self._whisper_modell.transcribe(
                    audio,
                    language="de",
                    beam_size=3,
                    vad_filter=True,          # Stille filtern
                    vad_parameters={"min_silence_duration_ms": 500},
                )
                return " ".join(s.text.strip() for s in segmente).strip()

            text = await loop.run_in_executor(None, _transkribiere)
            log.debug("STT: %s", text[:100])
            return text or None
        except Exception as e:
            log.error("Transkriptions-Fehler: %s", e)
            return None

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "tts_edge":      self._edge_verfuegbar,
            "tts_pyttsx3":   self._ist_windows,
            "stt_whisper":   self._whisper_verfuegbar,
            "stimme":        self.EDGE_STIMME,
            "whisper_modell":self.WHISPER_MODELL,
        }

    def installationshinweis(self) -> str:
        """Gibt Installationshinweise zurück wenn Pakete fehlen."""
        fehlend = []
        if not self._edge_verfuegbar:
            fehlend.append("pip install edge-tts")
        if not self._whisper_verfuegbar:
            fehlend.append("pip install faster-whisper")
        try:
            import sounddevice  # noqa
        except (ImportError, OSError):
            fehlend.append("pip install sounddevice")
        try:
            import pygame  # noqa
        except ImportError:
            fehlend.append("pip install pygame")

        if fehlend:
            return "Fehlende Voice-Pakete:\n" + "\n".join(f"  {p}" for p in fehlend)
        return "Alle Voice-Pakete installiert"
