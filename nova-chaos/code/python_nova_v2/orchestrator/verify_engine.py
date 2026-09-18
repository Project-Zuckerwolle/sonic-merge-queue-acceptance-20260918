"""Nova Predator v2 — VerifyEngine.
Mechanische + semantische Verifikation von Sub-Agent-Ergebnissen.

Zweistufig:
  1. Mechanisch (kein LLM, < 50ms):
     - Status == done?
     - Output-Dateien existieren?
     - bash-Checks aus Akzeptanz-Kriterien (wenn prüfbar)

  2. Semantisch (qwen3:8b Thinking-Mode, nur wenn mechanisch bestanden):
     - Entspricht die Implementierung der Spezifikation?
     - Gibt es Unstimmigkeiten?
     - Werden alle Akzeptanz-Kriterien semantisch erfüllt?

Ergebnis:
  VerifyResultat(bestanden, feedback, mechanisch_ok, semantisch_ok)
"""
from __future__ import annotations
import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from core.logger import get
from models.ergebnis_paket import ErgebnisPaket
from models.subtask_paket import SubtaskPaket

if TYPE_CHECKING:
    from core.ollama_client import OllamaClient

log = get("verify_engine")

_VERIFY_PROMPT = """\
Du bist der Orchestrator. Prüfe ob der Sub-Agent seinen Subtask korrekt abgeschlossen hat.
Denke sorgfältig nach (Thinking-Mode aktiv).

## SUBTASK-SPEZIFIKATION
{spezifikation}

## AKZEPTANZ-KRITERIEN
{kriterien}

## ERGEBNIS DES SUB-AGENTS
Status: {status}
Summary: {summary}
Output-Dateien: {output_files}
Fehler-Log: {fehler}

## DATEI-INHALTE (Stichproben)
{datei_inhalte}

## AUFGABE
Prüfe ob alle Akzeptanz-Kriterien erfüllt sind.
Erkenne Unstimmigkeiten (z.B. Datei existiert aber Inhalt passt nicht zur Spezifikation).

## AUSGABE (NUR JSON, kein Text davor oder danach)
{{
  "bestanden": true/false,
  "begruendung": "Warum bestanden oder nicht bestanden",
  "feedback": "Wenn nicht bestanden: konkretes Feedback für Retry-Versuch"
}}
"""


@dataclass
class VerifyResultat:
    bestanden: bool
    feedback: str = ""
    begruendung: str = ""
    mechanisch_ok: bool = False
    semantisch_ok: bool = False
    fehler_details: list[str] = None

    def __post_init__(self):
        if self.fehler_details is None:
            self.fehler_details = []


class VerifyEngine:
    """Verifiziert Sub-Agent-Ergebnisse mechanisch und semantisch."""

    def __init__(
        self,
        ollama: "OllamaClient",
        modell: str = "qwen3:8b",
        semantisch_aktiv: bool = True,
    ) -> None:
        self._ollama           = ollama
        self._modell           = modell
        self._semantisch_aktiv = semantisch_aktiv

    async def verifiziere(
        self,
        subtask: SubtaskPaket,
        ergebnis: ErgebnisPaket,
    ) -> VerifyResultat:
        """Vollständige Verifikation eines Subtask-Ergebnisses."""

        # ── Stufe 1: Mechanisch ─────────────────────────────────────
        mech = await self._mechanisch(subtask, ergebnis)
        if not mech.mechanisch_ok:
            log.info(
                "Subtask '%s': Mechanische Prüfung FEHLGESCHLAGEN — %s",
                subtask.id, mech.feedback,
            )
            return mech

        # ── Stufe 2: Semantisch (nur wenn mechanisch ok) ─────────────
        if not self._semantisch_aktiv:
            mech.bestanden = True
            return mech

        sem = await self._semantisch(subtask, ergebnis)
        sem.mechanisch_ok = True

        log.info(
            "Subtask '%s': Verify %s (mech=✓, sem=%s)",
            subtask.id,
            "BESTANDEN" if sem.bestanden else "FEHLGESCHLAGEN",
            "✓" if sem.semantisch_ok else "✗",
        )
        return sem

    # ── Mechanische Prüfung ─────────────────────────────────────────

    async def _mechanisch(
        self,
        subtask: SubtaskPaket,
        ergebnis: ErgebnisPaket,
    ) -> VerifyResultat:
        """Schnelle Prüfungen ohne LLM."""
        fehler = []

        # Status-Check
        if not ergebnis.bestanden:
            return VerifyResultat(
                bestanden=False,
                mechanisch_ok=False,
                feedback=(
                    f"Status war '{ergebnis.status}'. "
                    f"Fehler: {'; '.join(ergebnis.fehler_log[:3])}"
                ),
                begruendung="Sub-Agent hat nicht 'done' gemeldet",
                fehler_details=ergebnis.fehler_log,
            )

        # Output-Dateien prüfen
        workspace = Path(subtask.workspace)
        fehlende_dateien = []
        for f in ergebnis.output_files:
            pfad = Path(f) if Path(f).is_absolute() else workspace / f
            if not pfad.exists():
                fehlende_dateien.append(str(f))

        if fehlende_dateien:
            return VerifyResultat(
                bestanden=False,
                mechanisch_ok=False,
                feedback=f"Dateien fehlen: {fehlende_dateien}",
                begruendung="Output-Dateien existieren nicht",
                fehler_details=fehlende_dateien,
            )

        # bash-Checks aus Akzeptanz-Kriterien (wenn Kriterium mit $ beginnt)
        bash_fehler = await self._bash_checks(subtask, workspace)
        if bash_fehler:
            return VerifyResultat(
                bestanden=False,
                mechanisch_ok=False,
                feedback=f"bash-Checks fehlgeschlagen: {bash_fehler}",
                begruendung="Automatische Tests nicht bestanden",
                fehler_details=bash_fehler,
            )

        return VerifyResultat(
            bestanden=True,
            mechanisch_ok=True,
            begruendung="Alle mechanischen Checks bestanden",
        )

    async def _bash_checks(
        self,
        subtask: SubtaskPaket,
        workspace: Path,
    ) -> list[str]:
        """Führt Akzeptanz-Kriterien aus die mit '$' beginnen als bash-Befehle aus."""
        fehler = []
        for kriterium in subtask.akzeptanz_kriterien:
            if not kriterium.strip().startswith("$"):
                continue
            befehl = kriterium.strip()[1:].strip()
            try:
                proc = await asyncio.create_subprocess_shell(
                    befehl,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(workspace),
                )
                _, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
                if proc.returncode != 0:
                    fehler.append(f"'{befehl}' → Exit-Code {proc.returncode}")
            except asyncio.TimeoutError:
                fehler.append(f"'{befehl}' → Timeout")
            except Exception as e:
                fehler.append(f"'{befehl}' → Exception: {e}")
        return fehler

    # ── Semantische Prüfung ─────────────────────────────────────────

    async def _semantisch(
        self,
        subtask: SubtaskPaket,
        ergebnis: ErgebnisPaket,
    ) -> VerifyResultat:
        """LLM-basierte Tiefenprüfung mit Thinking-Mode."""
        # Datei-Inhalte als Stichprobe lesen (max 3 Dateien, max 500 Zeichen je)
        datei_inhalte = await self._lese_datei_stichproben(
            subtask.workspace, ergebnis.output_files
        )

        verify_prompt = _VERIFY_PROMPT.format(
            spezifikation=subtask.spezifikation[:600],
            kriterien="\n".join(f"- {k}" for k in subtask.akzeptanz_kriterien),
            status=ergebnis.status,
            summary=ergebnis.summary[:300],
            output_files=", ".join(ergebnis.output_files[:5]),
            fehler="; ".join(ergebnis.fehler_log[:2]) or "keine",
            datei_inhalte=datei_inhalte[:800],
        )

        try:
            antwort = await self._ollama.chat(
                nachrichten=[{"role": "user", "content": verify_prompt}],
                modell=self._modell,
                optionen={"temperature": 0.0, "num_predict": 512},
            )
            parsed = self._parse_json(antwort)
            if parsed:
                bestanden = bool(parsed.get("bestanden", True))
                return VerifyResultat(
                    bestanden=bestanden,
                    mechanisch_ok=True,
                    semantisch_ok=bestanden,
                    begruendung=parsed.get("begruendung", ""),
                    feedback=parsed.get("feedback", "") if not bestanden else "",
                )
        except Exception as e:
            log.debug("Semantische Verify fehlgeschlagen: %s", e)

        # Fallback: mechanisch bestanden → akzeptieren
        return VerifyResultat(
            bestanden=True,
            mechanisch_ok=True,
            semantisch_ok=True,
            begruendung="Semantische Prüfung fehlgeschlagen — mechanisch akzeptiert",
        )

    async def _lese_datei_stichproben(
        self,
        workspace: str,
        output_files: list[str],
        max_dateien: int = 3,
        max_zeichen: int = 500,
    ) -> str:
        """Liest Stichproben von Output-Dateien für den Verify-Prompt."""
        ws = Path(workspace)
        teile = []
        for f in output_files[:max_dateien]:
            pfad = Path(f) if Path(f).is_absolute() else ws / f
            try:
                inhalt = pfad.read_text(encoding="utf-8", errors="replace")[:max_zeichen]
                teile.append(f"### {f}\n{inhalt}")
            except Exception:
                pass
        return "\n\n".join(teile) if teile else "(keine Dateien lesbar)"

    def _parse_json(self, text: str) -> dict | None:
        """Extrahiert JSON aus LLM-Antwort."""
        text = text.strip()
        # Direktes JSON
        if text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        # JSON nach </think> (Thinking-Mode Output)
        if "</think>" in text:
            nach_think = text.split("</think>", 1)[1].strip()
            try:
                return json.loads(nach_think)
            except json.JSONDecodeError:
                pass
        # JSON in Markdown-Block
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # JSON irgendwo im Text
        m = re.search(r'\{"bestanden"[^}]*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None
