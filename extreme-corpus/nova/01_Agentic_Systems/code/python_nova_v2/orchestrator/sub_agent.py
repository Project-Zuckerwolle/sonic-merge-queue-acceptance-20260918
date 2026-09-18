"""Nova Predator v2 — SubAgent.
Autonomer Agent der einen Subtask vom Orchestrator ausführt.

Bekommt:
  - SubtaskPaket: vollständige Spezifikation, erlaubte Tools, Memory-Slice
  - OllamaClient: für LLM-Calls mit dem zugewiesenen Modell

Führt aus:
  - ReAct-Loop: Thought → Action → Observation (bis FINISH oder max_iterations)
  - LoopGuard: semantische Stagnation + max_iterations + max_errors
  - Schreibt Outputs in subtask.workspace

Gibt zurück:
  - ErgebnisPaket: status, output_files, summary, brain_updates, fehler_log
"""
from __future__ import annotations
import asyncio
import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.logger import get
from models.ergebnis_paket import ErgebnisPaket
from models.subtask_paket import SubtaskPaket

if TYPE_CHECKING:
    from core.ollama_client import OllamaClient

log = get("sub_agent")

_SYSTEM_PROMPT_TEMPLATE = """\
Du bist ein autonomer Sub-Agent von Nova Predator.
Du hast GENAU EINE Aufgabe. Arbeite sie vollständig ab.

## DEIN AUFTRAG
{ziel}

## SPEZIFIKATION
{spezifikation}

## KONTEXT
{kontext}

## ERLAUBTE TOOLS
{tools_liste}

## AKZEPTANZ-KRITERIEN (müssen alle erfüllt sein)
{kriterien}

## AUSGABE-FORMAT (strikt einhalten — nur JSON, kein Text davor/danach)
{{
  "thought": "Dein Gedankengang",
  "action": "tool_name",
  "action_input": {{...}}
}}
ODER bei Abschluss:
{{
  "thought": "Alle Kriterien erfüllt weil...",
  "action": "FINISH",
  "action_input": {{"summary": "Was wurde gebaut", "output_files": ["datei1.cpp", ...]}}
}}

## REGELN
- Prüfe nach jeder Action ob die Akzeptanz-Kriterien erfüllt sind
- Wenn ein Tool dreimal denselben Fehler gibt: anderes Vorgehen wählen
- FINISH erst wenn alle Akzeptanz-Kriterien erfüllt sind
- Schreibe alle erzeugten Dateien in: {workspace}
"""

_MAX_OBSERVATIONS_FÜR_STAGNATION = 3
_STAGNATIONS_THRESHOLD = 0.85  # Similarity für Stagnation


class SubAgent:
    """Autonomer Sub-Agent für einen einzelnen Subtask."""

    def __init__(
        self,
        subtask: SubtaskPaket,
        ollama: "OllamaClient",
    ) -> None:
        self.subtask = subtask
        self._ollama = ollama
        self._observations: list[str] = []
        self._fehler: list[str] = []
        self._output_files: list[str] = []
        self._brain_updates: list[dict] = []
        self._iterationen = 0

    async def ausfuehren(self) -> ErgebnisPaket:
        """Führt den Subtask autonom aus."""
        log.info(
            "SubAgent startet: '%s' (Modell: %s, max: %d iter)",
            self.subtask.titel, self.subtask.modell, self.subtask.max_iterationen,
        )
        workspace = Path(self.subtask.workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            ziel=self.subtask.ziel,
            spezifikation=self.subtask.spezifikation,
            kontext=self.subtask.kontext or "Kein zusätzlicher Kontext.",
            tools_liste="\n".join(f"- {t}" for t in self.subtask.tools_erlaubt),
            kriterien="\n".join(f"- {k}" for k in self.subtask.akzeptanz_kriterien),
            workspace=self.subtask.workspace,
        )

        nachrichten: list[dict] = []
        user_start = (
            f"Starte jetzt mit dem Subtask '{self.subtask.titel}'.\n"
            f"Workspace: {self.subtask.workspace}\n"
            f"Arbeite autonom bis alle Akzeptanz-Kriterien erfüllt sind."
        )
        nachrichten.append({"role": "user", "content": user_start})

        for iteration in range(self.subtask.max_iterationen):
            self._iterationen = iteration + 1

            try:
                antwort_roh = await self._ollama.chat(
                    nachrichten=nachrichten,
                    modell=self.subtask.modell,
                    system=system_prompt,
                    optionen={"temperature": 0.1, "num_predict": 2048},
                )
            except Exception as e:
                self._fehler.append(f"LLM-Fehler in Iteration {iteration}: {e}")
                log.error("SubAgent LLM-Fehler: %s", e)
                break

            nachrichten.append({"role": "assistant", "content": antwort_roh})

            # JSON parsen
            parsed = self._parse_json(antwort_roh)
            if not parsed:
                self._fehler.append(f"Iteration {iteration}: Kein valides JSON")
                nachrichten.append({
                    "role": "user",
                    "content": "FEHLER: Antworte NUR mit JSON, kein Text davor oder danach."
                })
                continue

            thought   = parsed.get("thought", "")
            action    = parsed.get("action", "")
            action_input = parsed.get("action_input", {})

            log.debug("Iteration %d: action=%s", iteration, action)

            # FINISH
            if action == "FINISH":
                summary = action_input.get("summary", "Subtask abgeschlossen")
                output_files = action_input.get("output_files", [])
                self._output_files.extend(output_files)
                log.info("SubAgent FINISH: '%s' in %d Iterationen", self.subtask.id, iteration+1)
                return ErgebnisPaket(
                    subtask_id=self.subtask.id,
                    status="done",
                    output_files=list(set(self._output_files)),
                    summary=summary,
                    fehler_log=self._fehler,
                    brain_updates=self._brain_updates,
                    iterationen_genutzt=self._iterationen,
                )

            # Tool ausführen
            observation = await self._tool_ausfuehren(action, action_input, workspace)
            self._observations.append(observation)

            # Stagnations-Check
            if self._ist_stagnierend():
                stagnation_msg = (
                    "HINWEIS: Deine letzten Beobachtungen waren sehr ähnlich. "
                    "Wähle einen anderen Ansatz oder prüfe die Akzeptanz-Kriterien."
                )
                nachrichten.append({"role": "user", "content": f"Observation: {observation}\n\n{stagnation_msg}"})
            else:
                nachrichten.append({"role": "user", "content": f"Observation: {observation}"})

        # max_iterations erreicht ohne FINISH
        log.warning("SubAgent '%s': max_iterations erreicht", self.subtask.id)
        return ErgebnisPaket(
            subtask_id=self.subtask.id,
            status="partial",
            output_files=list(set(self._output_files)),
            summary=f"Max Iterationen ({self.subtask.max_iterationen}) erreicht ohne Abschluss",
            fehler_log=self._fehler,
            brain_updates=self._brain_updates,
            iterationen_genutzt=self._iterationen,
            partial_done=self._output_files,
            partial_missing=[k for k in self.subtask.akzeptanz_kriterien],
        )

    async def _tool_ausfuehren(
        self,
        tool: str,
        args: dict,
        workspace: Path,
    ) -> str:
        """Führt ein erlaubtes Tool aus."""
        # Boundary-Check: nur erlaubte Tools
        if self.subtask.tools_erlaubt and tool not in self.subtask.tools_erlaubt:
            return f"FEHLER: Tool '{tool}' ist nicht erlaubt. Erlaubt: {self.subtask.tools_erlaubt}"

        try:
            if tool == "datei_schreiben":
                pfad = workspace / args.get("pfad", "output.txt")
                inhalt = args.get("inhalt", "")
                pfad.parent.mkdir(parents=True, exist_ok=True)
                pfad.write_text(inhalt, encoding="utf-8")
                self._output_files.append(str(pfad.relative_to(workspace)))
                return f"Datei '{pfad}' geschrieben ({len(inhalt)} Zeichen)"

            elif tool == "datei_lesen":
                pfad = Path(args.get("pfad", ""))
                if not pfad.is_absolute():
                    pfad = workspace / pfad
                if pfad.exists():
                    return pfad.read_text(encoding="utf-8")[:4000]
                return f"FEHLER: Datei '{pfad}' existiert nicht"

            elif tool == "notiz":
                inhalt = args.get("inhalt", "")
                self._brain_updates.append({"inhalt": inhalt, "typ": "fakt"})
                return f"Notiz gespeichert: {inhalt[:80]}"

            elif tool == "bash":
                # Einfache Bash-Ausführung im workspace
                befehl = args.get("befehl", "")
                if not befehl:
                    return "FEHLER: Kein Befehl angegeben"
                proc = await asyncio.create_subprocess_shell(
                    befehl,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(workspace),
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=30
                )
                output = stdout.decode("utf-8", errors="replace")
                errors = stderr.decode("utf-8", errors="replace")
                if errors:
                    self._fehler.append(f"bash stderr: {errors[:200]}")
                return (output or errors or "(kein Output)")[:2000]

            else:
                return f"FEHLER: Tool '{tool}' nicht implementiert"

        except asyncio.TimeoutError:
            return f"FEHLER: Tool '{tool}' Timeout nach 30s"
        except Exception as e:
            err = f"FEHLER: Tool '{tool}' Exception: {e}"
            self._fehler.append(err)
            return err

    def _parse_json(self, text: str) -> dict | None:
        """Extrahiert JSON aus LLM-Antwort."""
        text = text.strip()
        # Direkt JSON
        if text.startswith("{"):
            try:
                return json.loads(text)
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
        m = re.search(r"\{[^{}]*\"action\"[^{}]*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None

    def _ist_stagnierend(self) -> bool:
        """Prüft ob die letzten N Observations semantisch ähnlich sind."""
        if len(self._observations) < _MAX_OBSERVATIONS_FÜR_STAGNATION:
            return False
        letzte = self._observations[-_MAX_OBSERVATIONS_FÜR_STAGNATION:]
        # Einfacher Text-Overlap als Proxy (kein Embedding nötig hier)
        for i in range(len(letzte) - 1):
            w1 = set(letzte[i].lower().split())
            w2 = set(letzte[i+1].lower().split())
            if not w1 or not w2:
                continue
            overlap = len(w1 & w2) / min(len(w1), len(w2))
            if overlap < _STAGNATIONS_THRESHOLD:
                return False
        return True
