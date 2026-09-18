"""Nova Predator v2 — ModelCatalog.
Ersetzt ModelRouter. VRAM-aware Tier-System mit Cascade-Escalation.

Tiers (für RX 7900 XT, 20 GB VRAM):
  T0 nano   qwen2.5:3b        ~2.0 GB  Brain, Klassifikation
  T1 chat   gemma4:e4b        ~4.0 GB  Normaler Chat
  T2 think  qwen3:8b          ~5.0 GB  Orchestrator, Planung, Verify
  T3 code   codestral:22b    ~13.0 GB  Code schreiben/debuggen
  T4 embed  mxbai-embed-large ~0.3 GB  Embeddings (immer geladen)

VRAM-Regel: T2 + T3 niemals gleichzeitig (14 GB > 20 GB mit T4+T0).
            OrchestratorGateway verwaltet Handoff.

Cascade-Escalation:
  Klassifikation immer mit T0 → wenn Konfidenz < 0.7 → T1 → T2
  Code-Task → direkt T3
  Orchestrator → T2 (Thinking-Mode ON)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

from core.logger import get

log = get("model_catalog")

Slot = Literal["nano", "chat", "think", "code", "embed"]

# Tasks: was kann dieses Modell
TASK_KLASSIFIKATION = "klassifikation"
TASK_CHAT           = "chat"
TASK_PLANUNG        = "planung"
TASK_CODE           = "code"
TASK_EMBED          = "embed"
TASK_EXTRAKTION     = "extraktion"
TASK_VERIFY         = "verify"
TASK_ORCHESTRATOR   = "orchestrator"


@dataclass
class ModelEntry:
    slot:             Slot
    name:             str
    tier:             int
    vram_gb:          float
    ctx_len:          int
    quantization:     str
    tasks:            list[str]
    can_stream:       bool    = True
    thinking_mode:    bool    = False   # qwen3 unterstützt /think /nothink
    fallback_slot:    Slot | None = None
    warm:             bool    = False   # Laufzeit-Status: im VRAM geladen


# ── Standard-Catalog für RX 7900 XT ──────────────────────────────────────────
_DEFAULT_ENTRIES: list[ModelEntry] = [
    ModelEntry(
        slot="nano", name="qwen2.5:3b", tier=0, vram_gb=2.0,
        ctx_len=32768, quantization="Q4_K_M",
        tasks=[TASK_KLASSIFIKATION, TASK_EXTRAKTION, TASK_CHAT],
        fallback_slot=None,
    ),
    ModelEntry(
        slot="chat", name="gemma4:e4b", tier=1, vram_gb=4.0,
        ctx_len=32768, quantization="Q4_K_M",
        tasks=[TASK_CHAT, TASK_KLASSIFIKATION],
        fallback_slot="nano",
    ),
    ModelEntry(
        slot="think", name="qwen3:8b", tier=2, vram_gb=5.0,
        ctx_len=32768, quantization="Q4_K_M",
        tasks=[TASK_PLANUNG, TASK_ORCHESTRATOR, TASK_VERIFY, TASK_CHAT, TASK_CODE],
        thinking_mode=True,
        fallback_slot="chat",
    ),
    ModelEntry(
        slot="code", name="codestral:22b", tier=3, vram_gb=13.0,
        ctx_len=32768, quantization="Q4_K_M",
        tasks=[TASK_CODE, TASK_EXTRAKTION],
        fallback_slot="think",
    ),
    ModelEntry(
        slot="embed", name="mxbai-embed-large", tier=4, vram_gb=0.3,
        ctx_len=8192, quantization="FP16",
        tasks=[TASK_EMBED],
        can_stream=False,
        fallback_slot=None,
    ),
]


class ModelCatalog:
    """VRAM-aware Modell-Verwaltung mit Tier-Routing und Fallback-Chain.

    Rückwärts-kompatibel mit ModelRouter.waehle(typ) über .waehle()-Methode.
    """

    def __init__(self, eintraege: list[ModelEntry] | None = None) -> None:
        self._eintraege: dict[Slot, ModelEntry] = {}
        for e in (eintraege or _DEFAULT_ENTRIES):
            self._eintraege[e.slot] = e

        # Rückwärts-Kompatibilität: ModelRouter-Slot-Namen → Catalog-Slots
        self._compat_map: dict[str, Slot] = {
            "chat":    "chat",
            "brain":   "nano",
            "embed":   "embed",
            "schnell": "chat",
            "apex":    "think",     # Alter Apex-Slot → think
            "coder":   "code",
            "orchestrator": "think",
        }

    # ── Slot-Zugriff ────────────────────────────────────────────────────

    def get(self, slot: Slot) -> ModelEntry | None:
        return self._eintraege.get(slot)

    def waehle(self, typ: str) -> str:
        """Rückwärts-kompatibel mit ModelRouter.waehle(typ).
        Gibt Modellnamen zurück.
        """
        slot = self._compat_map.get(typ, "chat")
        entry = self._eintraege.get(slot)
        name = entry.name if entry else "gemma4:e4b"
        log.debug("ModelCatalog.waehle('%s') → slot=%s → %s", typ, slot, name)
        return name

    # ── Task-basiertes Routing ───────────────────────────────────────

    def waehle_fuer_task(
        self,
        task: str,
        vram_verfuegbar_gb: float = 20.0,
    ) -> ModelEntry:
        """Wählt niedrigsten Tier der den Task unterstützt und VRAM-Budget passt.

        Cascade: T0 → T1 → T2 → T3.
        Wenn kein Modell passt: Fallback auf chat.
        """
        kandidaten = [
            e for e in sorted(self._eintraege.values(), key=lambda x: x.tier)
            if task in e.tasks and e.vram_gb <= vram_verfuegbar_gb
        ]
        if kandidaten:
            gewählt = kandidaten[0]
            log.debug(
                "Task '%s': Slot=%s Modell=%s VRAM=%.1fGB",
                task, gewählt.slot, gewählt.name, gewählt.vram_gb,
            )
            return gewählt
        # Fallback
        fallback = self._eintraege.get("chat")
        log.warning("Kein Modell für Task '%s' bei %.1f GB VRAM, Fallback: chat", task, vram_verfuegbar_gb)
        return fallback  # type: ignore[return-value]

    def waehle_fuer_subtask(self, subtask_beschreibung: str) -> ModelEntry:
        """Capability-basierte Auswahl für A2A Sub-Agents.

        Erkennt Subtask-Typ aus Beschreibung und wählt optimales Modell.
        """
        desc = subtask_beschreibung.lower()
        code_keywords = {
            "code", "implementier", "schreib", "funktion", "klasse",
            "script", "bug", "debug", "compilier", "c++", "python",
        }
        logic_keywords = {
            "plan", "design", "architektur", "analysi",
            "review", "vergleich", "erkläre", "reasoning",
        }

        if any(kw in desc for kw in code_keywords):
            return self._eintraege.get("code", self._eintraege["think"])
        if any(kw in desc for kw in logic_keywords):
            return self._eintraege.get("think", self._eintraege["chat"])
        # Einfacher Subtask
        return self._eintraege.get("chat", self._eintraege["nano"])

    # ── Fallback-Chain ───────────────────────────────────────────────

    def fallback_chain(self, slot: Slot) -> list[str]:
        """Gibt geordnete Fallback-Kette als Modellnamen zurück."""
        kette: list[str] = []
        aktuell: Slot | None = slot
        besucht: set[Slot] = set()
        while aktuell and aktuell not in besucht:
            besucht.add(aktuell)
            entry = self._eintraege.get(aktuell)
            if entry:
                kette.append(entry.name)
            aktuell = entry.fallback_slot if entry else None
        return kette

    # ── VRAM-Management ──────────────────────────────────────────────

    def vram_gesamt(self) -> float:
        """Gibt geschätzten VRAM-Bedarf aller warmen (geladenen) Modelle zurück."""
        return sum(e.vram_gb for e in self._eintraege.values() if e.warm)

    def kann_laden(self, slot: Slot, vram_gesamt_gb: float = 20.0) -> bool:
        """Prüft ob ein Modell geladen werden kann ohne VRAM zu überlasten."""
        entry = self._eintraege.get(slot)
        if not entry:
            return False
        aktuell = self.vram_gesamt()
        return (aktuell + entry.vram_gb) <= vram_gesamt_gb

    def mark_warm(self, slot: Slot, warm: bool = True) -> None:
        """Markiert ein Modell als geladen/entladen (Laufzeit-Status)."""
        if slot in self._eintraege:
            self._eintraege[slot].warm = warm

    # ── Info ─────────────────────────────────────────────────────────

    def alle(self) -> list[ModelEntry]:
        return list(self._eintraege.values())

    def als_dict(self) -> dict:
        return {
            slot: {
                "name": e.name,
                "tier": e.tier,
                "vram_gb": e.vram_gb,
                "tasks": e.tasks,
                "warm": e.warm,
                "thinking": e.thinking_mode,
            }
            for slot, e in self._eintraege.items()
        }

    # ── Rückwärts-Kompatibilität mit ModelRouter ─────────────────────

    def fuer_intent(self, intents: list[str]) -> str:
        """ModelRouter.fuer_intent() Kompatibilität."""
        if "code" in intents:
            return self.waehle("coder")
        return self.waehle("chat")

    def alle_modelle(self) -> dict[str, str]:
        """ModelRouter.alle() Kompatibilität."""
        return {slot: e.name for slot, e in self._eintraege.items()}
