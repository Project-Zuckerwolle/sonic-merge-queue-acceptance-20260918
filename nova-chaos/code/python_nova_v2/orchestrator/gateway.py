"""Nova Predator v2 — OrchestratorGateway.
Verwaltet den VRAM-Handoff zwischen Orchestrator und Sub-Agents.

Garantiert dass Orchestrator-Modell (qwen3:8b ~5GB) und Code-Sub-Agent
(codestral:22b ~13GB) NIE gleichzeitig im VRAM sind.

Handoff-Sequenz (Orchestrator → Sub-Agent):
  1. Orchestrator-State auf Disk speichern
  2. Orchestrator-Modell aus VRAM entladen
  3. Warten bis VRAM < 8 GB (T0 + T4 bleiben geladen: ~2.3 GB)
  4. Sub-Agent-Modell laden
  5. Sub-Agent spawnen
  → Sub-Agent läuft autonom
  6. Sub-Agent gibt ErgebnisPaket zurück
  7. Sub-Agent-Modell entladen
  8. Warten bis VRAM frei
  9. Orchestrator-Modell laden
  10. Orchestrator-State von Disk wiederherstellen

Crash-Recovery: State liegt auf Disk — bei Neustart kann weiterggemacht werden.
"""
from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from core.logger import get
from models.subtask_paket import SubtaskPaket
from models.ergebnis_paket import ErgebnisPaket
from models.orchestrator_state import OrchestratorState

if TYPE_CHECKING:
    from core.ollama_client import OllamaClient
    from core.model_catalog import ModelCatalog

log = get("orchestrator_gateway")

# VRAM-Zielwerte nach Entladen (T0 + T4 bleiben immer)
_VRAM_NACH_ORCHESTRATOR_ENTLADEN_GB = 8.0   # qwen3:8b weg → ~2.3 GB frei
_VRAM_NACH_AGENT_ENTLADEN_GB        = 8.0   # Modell weg → ~2.3 GB frei
_HANDOFF_TIMEOUT_S                  = 60    # Max. Wartezeit für VRAM-Freigabe


class VRAMHandoffError(Exception):
    """Wird geworfen wenn VRAM-Freigabe nach Timeout nicht erreicht wurde."""


class OrchestratorGateway:
    """Verwaltet VRAM-Handoff und State-Persistenz für A2A-System."""

    def __init__(
        self,
        ollama: "OllamaClient",
        model_catalog: "ModelCatalog",
        orchestrator_modell: str = "qwen3:8b",
        state_pfad: Path = Path("workspace_agent/orchestrator_state.json"),
    ) -> None:
        self._ollama              = ollama
        self._catalog             = model_catalog
        self._orchestrator_modell = orchestrator_modell
        self._state_pfad          = state_pfad
        self._aktueller_state: OrchestratorState | None = None

    # ── Haupt-Handoff API ────────────────────────────────────────────

    async def handoff_zu_agent(
        self,
        subtask: SubtaskPaket,
        state: OrchestratorState,
    ) -> ErgebnisPaket:
        """Vollständiger Handoff-Zyklus: Orchestrator entladen → Agent läuft → Orchestrator laden.

        Returns:
            ErgebnisPaket des Sub-Agents.
        Raises:
            VRAMHandoffError: wenn VRAM nicht rechtzeitig freigegeben wird.
        """
        log.info(
            "Handoff → Sub-Agent für Subtask '%s' (Modell: %s)",
            subtask.id, subtask.modell,
        )

        # ── Phase 1: State persistieren ──────────────────────────────
        await self._state_speichern(state)

        # ── Phase 2: Orchestrator entladen ───────────────────────────
        log.info("Entlade Orchestrator-Modell '%s'", self._orchestrator_modell)
        await self._ollama.modell_entladen(self._orchestrator_modell)
        self._catalog.mark_warm("think", False)

        frei = await self._ollama.vram_warten(
            ziel_gb=_VRAM_NACH_ORCHESTRATOR_ENTLADEN_GB,
            timeout_s=_HANDOFF_TIMEOUT_S,
        )
        if not frei:
            raise VRAMHandoffError(
                f"VRAM nach Orchestrator-Entladen nicht freigegeben "
                f"(Timeout: {_HANDOFF_TIMEOUT_S}s)"
            )

        # ── Phase 3: Sub-Agent-Modell laden ──────────────────────────
        log.info("Lade Sub-Agent-Modell '%s'", subtask.modell)
        await self._ollama.modell_laden(subtask.modell)
        slot = self._slot_fuer_modell(subtask.modell)
        if slot:
            self._catalog.mark_warm(slot, True)

        # ── Phase 4: Sub-Agent spawnen und ausführen ─────────────────
        from orchestrator.sub_agent_factory import SubAgentFactory
        agent = SubAgentFactory.spawn(subtask, self._ollama)
        ergebnis = await agent.ausfuehren()

        # ── Phase 5: Sub-Agent-Modell entladen ───────────────────────
        log.info("Entlade Sub-Agent-Modell '%s'", subtask.modell)
        await self._ollama.modell_entladen(subtask.modell)
        if slot:
            self._catalog.mark_warm(slot, False)

        frei = await self._ollama.vram_warten(
            ziel_gb=_VRAM_NACH_AGENT_ENTLADEN_GB,
            timeout_s=_HANDOFF_TIMEOUT_S,
        )
        if not frei:
            raise VRAMHandoffError(
                f"VRAM nach Agent-Entladen nicht freigegeben "
                f"(Timeout: {_HANDOFF_TIMEOUT_S}s)"
            )

        # ── Phase 6: Orchestrator neu laden ──────────────────────────
        log.info("Lade Orchestrator-Modell '%s' neu", self._orchestrator_modell)
        await self._ollama.modell_laden(self._orchestrator_modell)
        self._catalog.mark_warm("think", True)

        log.info(
            "Handoff abgeschlossen: Subtask '%s' → Status: %s",
            subtask.id, ergebnis.status,
        )
        return ergebnis

    # ── State-Management ─────────────────────────────────────────────

    async def _state_speichern(self, state: OrchestratorState) -> None:
        """Speichert State auf Disk (async-safe via to_thread)."""
        self._aktueller_state = state
        await asyncio.to_thread(state.speichern, self._state_pfad)
        log.debug("Orchestrator-State gespeichert: %s", self._state_pfad)

    async def state_laden(self) -> OrchestratorState | None:
        """Lädt State von Disk (für Crash-Recovery)."""
        state = await asyncio.to_thread(OrchestratorState.laden, self._state_pfad)
        if state:
            log.info(
                "Crash-Recovery: State geladen (%s Subtasks, Fortschritt: %s)",
                len(state.subtasks), state.fortschritt(),
            )
        return state

    async def state_loeschen(self) -> None:
        """Löscht State nach erfolgreichem Abschluss."""
        def _loeschen():
            p = Path(self._state_pfad)
            if p.exists():
                p.unlink()
        await asyncio.to_thread(_loeschen)
        log.debug("Orchestrator-State gelöscht")

    # ── Hilfsmethoden ────────────────────────────────────────────────

    def _slot_fuer_modell(self, modellname: str) -> str | None:
        """Findet den Catalog-Slot für einen Modellnamen."""
        for entry in self._catalog.alle():
            if entry.name == modellname:
                return entry.slot
        return None

    async def orchestrator_verfuegbar(self) -> bool:
        """Prüft ob das Orchestrator-Modell geladen ist."""
        geladene = await self._ollama.geladene_modelle()
        return any(
            self._orchestrator_modell in m.get("name", "")
            for m in geladene
        )
