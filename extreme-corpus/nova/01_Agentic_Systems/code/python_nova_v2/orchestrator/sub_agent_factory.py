"""Nova Predator v2 — SubAgentFactory.
Erstellt Sub-Agent-Instanzen on-demand für das A2A-System.
Sub-Agents sind ephemer: werden für einen Subtask erstellt, danach verworfen.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from core.logger import get
from orchestrator.sub_agent import SubAgent

if TYPE_CHECKING:
    from core.ollama_client import OllamaClient
    from models.subtask_paket import SubtaskPaket

log = get("sub_agent_factory")


class SubAgentFactory:
    """Factory für dynamisches Sub-Agent-Spawning."""

    @staticmethod
    def spawn(subtask: "SubtaskPaket", ollama: "OllamaClient") -> SubAgent:
        """Erstellt neuen Sub-Agent für den Subtask.

        Kein dauerhafter State — jeder Sub-Agent ist isoliert:
        - Eigenes Modell (vom Orchestrator gewählt)
        - Eigener Workspace
        - Eigener Memory-Slice
        - Eigener Tool-Set (nur explizit erlaubte Tools)
        """
        log.info("Spawn Sub-Agent für Subtask %s (Modell: %s)",
                 getattr(subtask, "id", "?"), getattr(subtask, "modell", "?"))
        return SubAgent(subtask=subtask, ollama=ollama)
