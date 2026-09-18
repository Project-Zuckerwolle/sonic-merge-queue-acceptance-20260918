"""Nova Predator v3.1 — Session.
Einziger öffentlicher Einstiegspunkt für das Memory-System (Layer 1).
Koordiniert WorkingMemory, EpisodicMemory, SessionFacts, SemanticMemory, ContextBuilder.

v3.1: Optionale Conversation-Anbindung — jeder Turn wird via Callback auch
im ConversationManager persistiert. Rückwärts-kompatibel: ohne Callback
läuft die Session wie vorher.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Awaitable

from core.logger import get
from memory.context_builder import ContextBuilder, KontextErgebnis
from memory.episodic_memory import EpisodicMemory
from memory.session_facts import SessionFacts, FactTyp
from memory.semantic_memory import SemanticMemory
from memory.working_memory import WorkingMemory

if TYPE_CHECKING:
    from core.brain_index import BrainIndex
    from core.brain_manager import BrainManager
    from core.persona import Persona
    from core.skill_registry import SkillResult

log = get("session")

# Callback-Typ: (conv_id, rolle, inhalt) → fire-and-forget
# Wird beim Turn aufgerufen wenn gesetzt. Muss ein Coroutine returnen.
TurnPersistCallback = Callable[[str, str, str], Awaitable[None]]


class Session:
    """
    Öffentliche API für das gesamte Memory-System.
    Alle anderen Layer greifen NUR über diese Klasse auf Memory zu.
    """

    def __init__(
        self,
        persona: "Persona",
        brain_manager: "BrainManager",
        brain_index: "BrainIndex",
        max_tokens: int = 8000,
        compress_schwelle: float = 0.85,
    ) -> None:
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory(
            max_tokens=max_tokens,
            compress_schwelle=compress_schwelle,
        )
        self.session_facts = SessionFacts()
        self.semantic = SemanticMemory(brain_manager, brain_index)
        self.context_builder = ContextBuilder(
            persona=persona,
            episodic=self.episodic,
            session_facts=self.session_facts,
            max_tokens=max_tokens,
        )
        self._persona = persona

        # v3.1: Conversation-Anbindung (optional)
        self.conversation_id: str | None = None
        self._turn_persist_callback: TurnPersistCallback | None = None
        log.debug("Session initialisiert")

    # ── v3.1: Conversation-Hooks ─────────────────────────────────────────────

    def set_conversation(
        self,
        conv_id: str,
        persist_callback: TurnPersistCallback | None = None,
    ) -> None:
        """Setzt das aktive Gespräch + optionalen Persist-Callback.

        Der Callback wird bei jedem speichere_turn() als fire-and-forget
        ausgeführt (kein await im Hot-Path). Signatur:
            async def callback(conv_id: str, rolle: str, inhalt: str) -> None
        """
        self.conversation_id = conv_id
        self._turn_persist_callback = persist_callback
        log.info("Session an Gespräch gebunden: %s", conv_id)

    def lade_history_aus_turns(self, turns: list[dict]) -> int:
        """Lädt Chat-History aus einer Turn-Liste (z.B. von ConversationManager).

        Resettet zuerst das EpisodicMemory damit keine alten Turns bleiben.
        Session-Facts und Working-Memory werden NICHT angefasst — die sind
        separat.

        Returns:
            Anzahl geladener Turns.
        """
        self.episodic.reset()
        geladen = 0
        for t in turns:
            if not isinstance(t, dict):
                continue
            rolle = t.get("rolle", "")
            inhalt = t.get("inhalt", "")
            if rolle and inhalt:
                self.episodic.add(rolle, inhalt)
                geladen += 1
        log.debug("History geladen: %d Turns", geladen)
        return geladen

    # ── Bestehende API (unverändert) ─────────────────────────────────────────

    async def baue_kontext(
        self,
        user_input: str,
        q_vec: list[float] | None = None,
        skill_ergebnisse: dict[str, "SkillResult"] | None = None,
    ) -> KontextErgebnis:
        """Hauptmethode: Baut vollständigen LLM-Kontext auf.
        Sucht Brain-Fakten, integriert Session-Fakten, baut Messages.
        """
        # Brain-Fakten via Hybrid Search (Vector + BM25 + Recency)
        brain_fakten: list[str] = []
        if q_vec or user_input:
            brain_fakten = await self.semantic.als_fakten_text(
                q_vec          = q_vec or [],
                query_text     = user_input,
                max_ergebnisse = 5,
            )
        brain_entry_ids = self.semantic.letzte_entry_ids

        # Kontext zusammenbauen
        kontext = self.context_builder.baue(
            user_input=user_input,
            brain_fakten=brain_fakten,
            skill_ergebnisse=skill_ergebnisse,
        )
        kontext.brain_entry_ids = brain_entry_ids

        # Prüfen ob History komprimiert werden muss
        if self.episodic.braucht_kompression():
            entfernt = self.episodic.kuerze(auf_prozent=0.4)
            log.info("Session History gekürzt: %d Turns entfernt", entfernt)

        return kontext

    def speichere_turn(self, user_input: str, antwort: str) -> None:
        """Speichert einen abgeschlossenen Turn in die History.

        v3.1: Wenn eine Conversation-ID gesetzt ist und ein Persist-Callback
        existiert, wird der Turn zusätzlich im ConversationManager persistiert
        (fire-and-forget via asyncio.create_task — blockiert nicht).
        """
        # EpisodicMemory (sync, in-memory)
        self.episodic.add("user", user_input)
        self.episodic.add("assistant", antwort)

        # v3.1: Optional in ConversationManager persistieren
        if self.conversation_id and self._turn_persist_callback:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                cb = self._turn_persist_callback
                conv_id = self.conversation_id

                # Wrapper der Exceptions loggt statt sie zu verlieren
                async def _safe_persist(rolle: str, inhalt: str) -> None:
                    try:
                        await cb(conv_id, rolle, inhalt)
                    except Exception as e:
                        log.error("Turn-Persist-Callback Fehler (%s): %s", rolle, e)

                loop.create_task(_safe_persist("user", user_input))
                loop.create_task(_safe_persist("assistant", antwort))
            except RuntimeError:
                # Kein laufender Loop (z.B. im Test) — überspringen
                log.debug("Turn-Persist ohne Loop übersprungen")

    def fact_hinzufuegen(self, text: str, typ: FactTyp = "fakt", konfidenz: float = 0.9) -> None:
        """Fügt einen verifizierten Fakt zur Session hinzu."""
        self.session_facts.add(text, typ, konfidenz)

    async def archivieren(self) -> Path:
        """Archiviert die Session und resettet History + Working Memory.

        Hinweis v3.1: Die neue Architektur persistiert Turns kontinuierlich
        via ConversationManager. archivieren() bleibt für Kompatibilität
        (alte /ws/chat Endpoint, /api/memory/archivieren).
        """
        datei = await self.episodic.archiviere()
        self.session_facts.reset()
        self.working.reset()
        return datei

    def reset(self) -> None:
        """Resettet Session ohne Archivierung (z.B. im Test)."""
        self.episodic.reset()
        self.session_facts.reset()
        self.working.reset()

    def status(self) -> dict:
        return {
            "episodic": self.episodic.status(),
            "session_facts": len(self.session_facts),
            "working": self.working.als_dict(),
            "conversation_id": self.conversation_id,  # v3.1
        }
