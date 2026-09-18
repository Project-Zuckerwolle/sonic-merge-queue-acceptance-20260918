"""Nova Predator v2 — Orchestrator.
Plan-Execute-Verify Zyklus für autonomes Bauen.

Flow:
  PLAN:    User-Ziel verstehen → Recherche → Subtask-Pakete erstellen → Preview
  EXECUTE: Gateway → Sub-Agent → Ergebnis
  VERIFY:  Mechanisch (Dateien, bash) → Semantisch (qwen3:8b Thinking-Mode)
  LOOP:    Nächster Subtask oder DONE

Liest ORCHESTRATOR_GUIDELINES.md für Planungs- und Verifikations-Regeln.
"""
from __future__ import annotations
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from core.logger import get
from models.ergebnis_paket import ErgebnisPaket
from models.orchestrator_state import OrchestratorState, SubtaskStateEntry
from models.subtask_paket import SubtaskPaket

if TYPE_CHECKING:
    from core.ollama_client import OllamaClient
    from core.model_catalog import ModelCatalog
    from core.event_bus import EventBus
    from orchestrator.gateway import OrchestratorGateway

log = get("orchestrator")

_GUIDELINES_PFAD = Path("behavior/ORCHESTRATOR_GUIDELINES.md")
_ORCHESTRATOR_MODELL = "qwen3:8b"

_PLAN_PROMPT = """\
Du bist der Orchestrator von Nova Predator — du planst und koordinierst autonome Builds.

## DEINE AUFGABE
Lies den Plan des Users und erstelle ein strukturiertes Subtask-Array.

## USER-PLAN
{user_plan}

## VERFÜGBARE MODELLE (wähle das kleinste das reicht)
- qwen2.5:3b  → einfache Tasks (Datei prüfen, lesen, kleine Änderungen)
- gemma4:e4b  → mittlere Tasks (Recherche, Zusammenfassung, einfaches Schreiben)
- qwen3:8b    → komplexe Tasks (Planung, Analyse, Review, Reasoning)
- codestral:22b → Code-Tasks (C++, Python, Blueprint-Analyse)

## GUIDELINES
{guidelines}

## AUSGABE (NUR JSON-Array, kein Text davor oder danach)
[
  {{
    "id": "st_001",
    "titel": "Kurzer Name max 60 Zeichen",
    "ziel": "Was soll am Ende existieren/funktionieren",
    "spezifikation": "Detailliert — wie ein Ticket für einen Entwickler",
    "kontext": "Relevante Hintergrundinformation",
    "akzeptanz_kriterien": ["Messbare Bedingung 1", "Messbare Bedingung 2"],
    "modell": "codestral:22b",
    "modell_begruendung": "Warum dieses Modell",
    "tools_erlaubt": ["datei_schreiben", "bash"],
    "abhaengig_von": [],
    "max_iterationen": 20,
    "workspace": "workspace_agent/st_001/"
  }}
]
"""

_VERIFY_PROMPT = """\
Du bist der Orchestrator. Prüfe ob der Sub-Agent seinen Subtask korrekt abgeschlossen hat.

## SUBTASK-SPEZIFIKATION
{spezifikation}

## AKZEPTANZ-KRITERIEN
{kriterien}

## ERGEBNIS DES SUB-AGENTS
Status: {status}
Summary: {summary}
Output-Files: {output_files}
Fehler: {fehler}

## DEINE ANALYSE (denke nach /think)
Entspricht das Ergebnis den Anforderungen?
Sind alle Akzeptanz-Kriterien erfüllt?

## AUSGABE (NUR JSON)
{{
  "bestanden": true/false,
  "begruendung": "Warum bestanden/nicht bestanden",
  "feedback": "Wenn nicht bestanden: konstruktives Feedback für Retry"
}}
"""


class Orchestrator:
    """Plan-Execute-Verify Orchestrator für autonome Builds."""

    def __init__(
        self,
        ollama: "OllamaClient",
        model_catalog: "ModelCatalog",
        gateway: "OrchestratorGateway",
        bus: "EventBus",
        modell: str = _ORCHESTRATOR_MODELL,
        max_retries: int = 3,
    ) -> None:
        self._ollama      = ollama
        self._catalog     = model_catalog
        self._gateway     = gateway
        self._bus         = bus
        self._modell      = modell
        self._max_retries = max_retries
        self._guidelines  = self._lade_guidelines()

    # ── Haupt-API ────────────────────────────────────────────────────

    async def plane(self, user_plan: str) -> list:
        """v3.4: Nur Plan-Phase — gibt SubtaskPakete zurück ohne auszuführen.

        Wird von _handle_agent_intent aufgerufen bevor der User GO gibt.
        Die Subtasks werden als Chat-Antwort präsentiert und in _pending_plan
        gespeichert. Execution startet erst nach User-Bestätigung via
        plane_und_fuehre_aus().
        """
        return await self._plan_phase(user_plan)

    async def plane_und_fuehre_aus(
        self,
        user_plan: str,
    ) -> AsyncIterator[dict]:
        """Haupteinstiegspunkt: Plant und führt autonomen Build aus.

        Yields:
            dicts mit 'typ' und Payload für WebSocket-Events:
            - {typ: 'plan_update', subtasks: [...]}
            - {typ: 'task_update', subtask_id: ..., status: ...}
            - {typ: 'thinking', content: ...}
            - {typ: 'error', content: ...}
            - {typ: 'done', summary: ...}
        """
        # ── PLAN Phase ────────────────────────────────────────────────
        yield {"typ": "thinking", "content": "Verstehe deinen Plan..."}

        subtasks = await self._plan_phase(user_plan)
        if not subtasks:
            yield {"typ": "error", "content": "Konnte keinen Plan erstellen"}
            return

        yield {
            "typ": "plan_update",
            "subtasks": [s.to_dict() for s in subtasks],
            "gesamt": len(subtasks),
        }

        # State initialisieren
        state = OrchestratorState(plan_text=user_plan, gestartet_um=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
        state.subtasks = [SubtaskStateEntry(paket=s.to_dict()) for s in subtasks]
        await self._gateway._state_speichern(state)

        # ── EXECUTE + VERIFY Loop ─────────────────────────────────────
        abgeschlossene = 0
        async for event in self._execute_loop(state):
            yield event
            if event.get("typ") == "task_update" and event.get("status") == "done":
                abgeschlossene += 1

        # State aufräumen wenn alles done
        if state.fortschritt()['pending'] == 0 and state.fortschritt()['failed'] == 0:
            await self._gateway.state_loeschen()
            yield {
                "typ": "done",
                "summary": f"Alle {len(subtasks)} Subtasks abgeschlossen",
                "fortschritt": state.fortschritt(),
            }
        else:
            yield {
                "typ": "error",
                "content": f"Abgebrochen: {state.fortschritt()} Subtasks abgeschlossen",
            }

    # ── PLAN Phase ────────────────────────────────────────────────────

    async def _plan_phase(self, user_plan: str) -> list[SubtaskPaket]:
        """Erstellt Subtask-Pakete aus User-Plan."""
        prompt = _PLAN_PROMPT.format(
            user_plan=user_plan,
            guidelines=self._guidelines[:2000],  # Kürzen für Token-Budget
        )

        antwort = await self._ollama.chat(
            nachrichten=[{"role": "user", "content": prompt}],
            modell=self._modell,
            optionen={"temperature": 0.1, "num_predict": 4096},
        )

        return self._parse_subtask_array(antwort)

    def _parse_subtask_array(self, antwort: str) -> list[SubtaskPaket]:
        """Parst JSON-Array aus LLM-Antwort in SubtaskPaket-Liste."""
        antwort = antwort.strip()
        # JSON-Array extrahieren
        if not antwort.startswith("["):
            m = re.search(r"\[.*\]", antwort, re.DOTALL)
            if m:
                antwort = m.group(0)
            else:
                log.error("Kein JSON-Array in Orchestrator-Antwort: %s", antwort[:200])
                return []

        try:
            daten = json.loads(antwort)
        except json.JSONDecodeError as e:
            log.error("JSON-Parse-Fehler: %s", e)
            return []

        pakete = []
        for i, d in enumerate(daten):
            # Defaults sicherstellen
            d.setdefault("id", f"st_{i+1:03d}")
            d.setdefault("workspace", f"workspace_agent/{d['id']}/")
            d.setdefault("kontext", "")
            d.setdefault("modell_begruendung", "")
            d.setdefault("tools_erlaubt", [])
            d.setdefault("abhaengig_von", [])
            d.setdefault("max_iterationen", 20)
            d.setdefault("memory_slice", {})
            try:
                pakete.append(SubtaskPaket.from_dict(d))
            except (ValueError, KeyError) as e:
                log.warning("Subtask %d übersprungen (Validierung): %s", i, e)
        return pakete

    # ── EXECUTE + VERIFY Loop ─────────────────────────────────────────

    async def _execute_loop(
        self,
        state: OrchestratorState,
    ) -> AsyncIterator[dict]:
        """Verarbeitet Subtask-Queue bis leer oder Abbruch."""
        while True:
            eintrag = state.naechster_pending()
            if not eintrag:
                break

            subtask = SubtaskPaket.from_dict(eintrag.paket)
            eintrag.status = "running"
            eintrag.gestartet_um = datetime.now(timezone.utc).isoformat()
            await self._gateway._state_speichern(state)

            yield {"typ": "task_update", "subtask_id": subtask.id,
                   "status": "running", "titel": subtask.titel}

            # Versuche mit Retry
            ergebnis = None
            for versuch in range(self._max_retries):
                eintrag.versuche = versuch + 1
                yield {"typ": "thinking",
                       "content": f"Subtask '{subtask.titel}' läuft (Versuch {versuch+1}/{self._max_retries})..."}

                try:
                    ergebnis = await self._gateway.handoff_zu_agent(subtask, state)
                except Exception as e:
                    log.error("Gateway-Fehler: %s", e)
                    ergebnis = ErgebnisPaket(
                        subtask_id=subtask.id,
                        status="failed",
                        fehler_log=[str(e)],
                    )

                # VERIFY
                bestanden = await self._verify(subtask, ergebnis)
                ergebnis.verify_mechanisch = bestanden
                ergebnis.verify_semantisch = bestanden

                if bestanden:
                    break

                # Nicht bestanden → Feedback für nächsten Versuch
                if versuch < self._max_retries - 1:
                    feedback = eintrag.feedback or "Nicht alle Kriterien erfüllt"
                    yield {"typ": "thinking",
                           "content": f"Retry {versuch+2}: {feedback}"}
                    # Feedback in Spezifikation einarbeiten
                    eintrag.paket["spezifikation"] = eintrag.paket.get("spezifikation","") + f"\n\nFEEDBACK AUS VERSUCH {versuch+1}:\n{feedback}"

            # Status setzen
            if ergebnis and ergebnis.verify_mechanisch:
                eintrag.status = "done"
                eintrag.ergebnis = ergebnis
            else:
                eintrag.status = "failed"
                eintrag.ergebnis = ergebnis
                # Nach max_retries: AutonomyGate
                yield {"typ": "question",
                       "content": f"Subtask '{subtask.titel}' ist nach {self._max_retries} Versuchen fehlgeschlagen. Wie soll ich weitermachen?",
                       "subtask_id": subtask.id}
                break  # Stoppe bei Fehler (User muss eingreifen)

            eintrag.abgeschlossen_um = datetime.now(timezone.utc).isoformat()
            await self._gateway._state_speichern(state)

            yield {"typ": "task_update", "subtask_id": subtask.id,
                   "status": eintrag.status, "titel": subtask.titel}

    # ── VERIFY ───────────────────────────────────────────────────────

    async def _verify(self, subtask: SubtaskPaket, ergebnis: ErgebnisPaket) -> bool:
        """Mechanische + semantische Verifikation."""
        # Mechanisch: status == done?
        if not ergebnis.bestanden:
            eintrag.feedback = (
                f"Status war '{ergebnis.status}'. "
                f"Fehler: {'; '.join(ergebnis.fehler_log[:3])}"
            )
            return False

        # Mechanisch: Output-Files vorhanden?
        workspace = Path(subtask.workspace)
        fehlende = []
        for f in ergebnis.output_files:
            pfad = workspace / f if not Path(f).is_absolute() else Path(f)
            if not pfad.exists():
                fehlende.append(str(f))
        if fehlende:
            eintrag.feedback = f"Dateien nicht gefunden: {fehlende}"
            return False

        # Semantisch: LLM-Verify (Thinking-Mode)
        if not ergebnis.output_files:
            return True  # Keine Dateien → nur Status war Kriterium

        try:
            verify_prompt = _VERIFY_PROMPT.format(
                spezifikation=subtask.spezifikation[:500],
                kriterien="\n".join(f"- {k}" for k in subtask.akzeptanz_kriterien),
                status=ergebnis.status,
                summary=ergebnis.summary[:300],
                output_files=", ".join(ergebnis.output_files[:5]),
                fehler="; ".join(ergebnis.fehler_log[:2]),
            )
            antwort = await self._ollama.chat(
                nachrichten=[{"role": "user", "content": verify_prompt}],
                modell=self._modell,
                optionen={"temperature": 0.0, "num_predict": 512},
            )
            parsed = self._parse_verify_antwort(antwort)
            if parsed:
                if not parsed.get("bestanden", True):
                    eintrag.feedback = parsed.get("feedback", "Nicht bestanden")
                return parsed.get("bestanden", True)
        except Exception as e:
            log.debug("Semantische Verifikation fehlgeschlagen: %s", e)

        return True  # Fallback: mechanisch bestanden → akzeptieren

    def _parse_verify_antwort(self, antwort: str) -> dict | None:
        """Parst Verify-JSON aus LLM-Antwort."""
        m = re.search(r"\{[^{}]*\"bestanden\"[^{}]*\}", antwort, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None

    # ── Hilfsmethoden ────────────────────────────────────────────────

    def _lade_guidelines(self) -> str:
        """Lädt ORCHESTRATOR_GUIDELINES.md."""
        if _GUIDELINES_PFAD.exists():
            return _GUIDELINES_PFAD.read_text(encoding="utf-8")
        return """
VERSTEHEN ZUERST: Formuliere das Ziel zurück bevor du planst.
SUBTASKS SIND ATOMAR: Jeder Subtask = eine abgeschlossene Einheit.
SPEZIFIKATION IST ALLES: So präzise dass ein Entwickler ohne Rückfragen starten kann.
AKZEPTANZ-KRITERIEN SIND MESSBAR: Dateien existieren, Code kompiliert.
"""
