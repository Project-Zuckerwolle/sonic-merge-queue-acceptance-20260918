"""Nova Predator v1 Layer 6 — Datenstrukturen.

ApexTask:    Ein vollständiger autonomer Auftrag mit ReAct-Loop, eigener Memory
             und Anbindung an Brain + Session des Haupt-Chats.
ReActStep:   Ein einzelner Denk-Handle-Beobacht-Zyklus.
TraceLog:    Persistente JSONL-Aufzeichnung aller Schritte für Debugging + Learning.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ApexStatus(Enum):
    PENDING    = "pending"
    PLANNING   = "planning"
    RUNNING    = "running"
    REFLECTING = "reflecting"
    REPLANNING = "replanning"
    DONE       = "done"
    FAILED     = "failed"
    ABORTED    = "aborted"


class ReActTyp(Enum):
    THOUGHT    = "thought"     # Denken: Was tue ich als nächstes?
    ACTION     = "action"      # Handeln: Tool aufrufen
    OBSERVATION= "observation" # Beobachten: Was hat das Ergebnis gebracht?
    REFLECTION = "reflection"  # Reflektieren: Bin ich auf dem richtigen Weg?
    REPLAN     = "replan"      # Neuer Plan weil alter nicht funktioniert
    MEMORY_FLUSH = "memory_flush"  # Kontext-Kompaktierung


@dataclass
class ReActStep:
    """Ein einzelner Schritt im ReAct-Loop."""
    id:        str   = field(default_factory=lambda: uuid.uuid4().hex[:8])
    typ:       ReActTyp = ReActTyp.THOUGHT
    inhalt:    str   = ""          # Text des Gedankens / der Beobachtung
    tool:      str   = ""          # Tool-Name wenn ACTION
    tool_args: dict  = field(default_factory=dict)
    tool_result: str = ""          # Ergebnis des Tool-Calls
    success:   bool  = True
    tokens:    int   = 0
    dauer_ms:  int   = 0
    ts:        float = field(default_factory=time.time)

    def als_dict(self) -> dict:
        return {
            "id":          self.id,
            "typ":         self.typ.value,
            "inhalt":      self.inhalt[:500],
            "tool":        self.tool,
            "tool_args":   self.tool_args,
            "tool_result": self.tool_result[:500],
            "success":     self.success,
            "dauer_ms":    self.dauer_ms,
            "ts":          self.ts,
        }


@dataclass
class ApexMemory:
    """Eigener Memory-Bereich für den Layer-6-Agent.

    Getrennt vom Chat-Memory, aber Wichtiges wird per Memory-Bridge
    in Brain + Session des Haupt-Chats eingetragen.

    scratchpad: Kurzzeit-Notizen für den laufenden Task (wird kompaktiert)
    facts:      Extrahierte Fakten (persistent im Task-Verzeichnis)
    summary:    Komprimierte Zusammenfassung nach Kontext-Kompaktierung
    """
    scratchpad: list[str]        = field(default_factory=list)
    facts:      list[str]        = field(default_factory=list)
    summary:    str              = ""
    token_schaetzung: int        = 0

    MAX_SCRATCHPAD_TOKENS = 3000

    def add_thought(self, text: str) -> None:
        self.scratchpad.append(f"[Gedanke] {text}")
        self.token_schaetzung += len(text) // 4

    def add_observation(self, text: str, limit: int = 2000) -> None:
        """limit kann erhöht werden für datei_lesen (voller Inhalt sichtbar)."""
        self.scratchpad.append(f"[Beobachtung] {text[:limit]}")
        self.token_schaetzung += min(len(text) // 4, 500)

    def add_fact(self, fact: str) -> None:
        if fact not in self.facts:
            self.facts.append(fact)

    def braucht_kompaktierung(self) -> bool:
        return self.token_schaetzung > self.MAX_SCRATCHPAD_TOKENS

    def kompaktiere(self, zusammenfassung: str) -> None:
        """Ersetzt Scratchpad durch Zusammenfassung — Fakten bleiben."""
        self.summary = zusammenfassung
        self.scratchpad = []
        self.token_schaetzung = len(zusammenfassung) // 4

    def als_kontext(self) -> str:
        """Baut Kontext-String für System-Prompt."""
        parts: list[str] = []
        if self.summary:
            parts.append(f"[Zusammenfassung bisheriger Arbeit]\n{self.summary}")
        if self.facts:
            parts.append("[Extrahierte Fakten]\n" + "\n".join(f"• {f}" for f in self.facts[-20:]))
        if self.scratchpad:
            parts.append("[Letzte Schritte]\n" + "\n".join(self.scratchpad[-12:]))
        return "\n\n".join(parts)


@dataclass
class ApexTask:
    """Ein vollständiger autonomer Auftrag."""
    id:          str
    aufgabe:     str
    status:      ApexStatus     = ApexStatus.PENDING
    schritte:    list[ReActStep]= field(default_factory=list)
    memory:      ApexMemory     = field(default_factory=ApexMemory)
    plan:        list[str]      = field(default_factory=list)     # Aktueller Plan als Schritte
    ergebnis:    str            = ""
    fehler:      str            = ""
    iteration:   int            = 0
    max_iter:    int            = 40
    workspace:   Path           = field(default=Path("workspace"))
    started_at:  float          = field(default_factory=time.time)
    done_at:     float          = 0.0
    # Was in Brain+Session eingetragen werden soll
    brain_eintraege: list[str]  = field(default_factory=list)

    def log_schritt(self, schritt: ReActStep) -> None:
        self.schritte.append(schritt)
        self.iteration += 1

    def letzter_output(self) -> str:
        """Letztes Tool-Ergebnis oder leerer String."""
        for s in reversed(self.schritte):
            if s.typ == ReActTyp.OBSERVATION and s.tool_result:
                return s.tool_result
        return ""

    def fortschritt_stagniert(self, fenster: int = 5) -> bool:
        """True wenn der Agent in einer Schleife steckt.

        Zwei Bedingungen:
        1. Alle letzten Actions fehlgeschlagen (success=False)
        2. Gleiche Action + gleiche Args 3x hintereinander (auch wenn success=True)
        """
        letzte = [s for s in self.schritte[-fenster:] if s.typ == ReActTyp.ACTION]
        if len(letzte) < 3:
            return False
        # Bedingung 1: alle gescheitert
        if all(not s.success for s in letzte):
            return True
        # Bedingung 2: gleiches Tool mit gleichen Args 3x wiederholt
        if len(letzte) >= 3:
            letzte3 = letzte[-3:]
            gleiche_tools = len(set(s.tool for s in letzte3)) == 1
            gleiche_args  = len(set(
                str(sorted(s.tool_args.items())) for s in letzte3
            )) == 1
            if gleiche_tools and gleiche_args:
                return True
        return False

    def zusammenfassung(self) -> str:
        done  = sum(1 for s in self.schritte if s.typ == ReActTyp.OBSERVATION and s.success)
        total = sum(1 for s in self.schritte if s.typ == ReActTyp.ACTION)
        dauer = int((self.done_at or time.time()) - self.started_at)
        return (f"{done}/{total} Actions erfolgreich · "
                f"{self.iteration} Iterationen · {dauer}s")


class TraceLog:
    """Schreibt jeden ReAct-Schritt in JSONL-Datei.

    Format: eine JSON-Zeile pro Schritt.
    Ermöglicht Debugging, Replay und spätere Prompt-Verbesserungen.
    """

    def __init__(self, workspace: Path, task_id: str) -> None:
        workspace.mkdir(parents=True, exist_ok=True)
        self._datei = workspace / f"{task_id}_trace.jsonl"

    def schreibe(self, schritt: ReActStep, task_id: str) -> None:
        """Schreibt Schritt in Trace-Datei (sync — läuft im Thread)."""
        try:
            eintrag = {"task_id": task_id, **schritt.als_dict()}
            with open(self._datei, "a", encoding="utf-8") as f:
                f.write(json.dumps(eintrag, ensure_ascii=False) + "\n")
        except Exception:
            pass  # Trace-Fehler niemals den Agent stoppen

    def lese_alle(self) -> list[dict]:
        """Liest alle Schritte aus Trace-Datei."""
        if not self._datei.exists():
            return []
        eintraege = []
        try:
            for zeile in self._datei.read_text(encoding="utf-8").splitlines():
                if zeile.strip():
                    eintraege.append(json.loads(zeile))
        except Exception:
            pass
        return eintraege
