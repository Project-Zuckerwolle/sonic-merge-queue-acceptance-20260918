"""Nova Predator v3.4 — IntentClassifier.

Ersetzt NanoParser.plan_trigger + NanoParser.komplex.
Einziger LLM-Call pro Chat-Turn der entscheidet:
  - route: "chat" | "agent"
  - komplex: bool   (Thinking-Mode für Chat-LLM)
  - agent_summary: str | None  (was Nova bauen soll, für Orchestrator)

Läuft parallel zum Kontext-Aufbau → kein Netto-Latenz-Overhead.
Bei Timeout oder Fehler: Fallback auf route="chat", komplex=False.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

from core.logger import get

log = get("intent_classifier")

_TIMEOUT_S = 3.0
_MODELL    = "qwen2.5:3b"

_SYSTEM = """\
Du klassifizierst Nutzeranfragen für ein lokales KI-System namens Nova.
Antworte NUR mit einem JSON-Objekt — kein Text davor oder danach, keine Markdown-Blöcke.

AGENT: Der User will etwas erschaffen das als Datei, Script, Code oder Projekt existiert.
       Typisch: mehrere abhängige Schritte, Output ist ein File oder Ordner-Struktur.
       Auch: "Ja" / "Mach" / "Loslegen" wenn Nova im vorherigen Turn einen Plan vorgestellt hat.

CHAT: Fragen, Erklärungen, Analysen, kurze Aktionen (Suche, Wetter, Übersetzung).
      Kein dauerhafter File-Output erwartet.

KOMPLEX: True wenn die Chat-Antwort tiefes Reasoning braucht —
         Vergleiche, Architekturen, mehrstufige Analysen, viele Teilfragen.
         Bei route=agent immer false.

Ausgabe-Schema:
{
  "route": "chat" | "agent",
  "komplex": true | false,
  "agent_summary": "<was gebaut werden soll>" | null
}"""


@dataclass
class IntentErgebnis:
    route: str          = "chat"   # "chat" | "agent"
    komplex: bool       = False
    agent_summary: str | None = None


class IntentClassifier:
    """LLM-basierter Intent-Classifier für Chat vs. Agent-Routing."""

    def __init__(self, ollama: object, modell: str = _MODELL) -> None:
        self._ollama  = ollama
        self._modell  = modell

    async def klassifiziere(
        self,
        user_input: str,
        letzte_turns: list[dict] | None = None,
    ) -> IntentErgebnis:
        """Klassifiziert user_input als 'chat' oder 'agent'.

        Args:
            user_input:   Aktuelle Nutzer-Nachricht.
            letzte_turns: Letzte ≤6 Turns als [{"role": ..., "content": ...}].
                          Gibt dem LLM Kontext um z.B. "Ja" als Plan-Bestätigung
                          zu erkennen wenn Nova vorher einen Plan präsentiert hat.

        Returns:
            IntentErgebnis mit route, komplex, agent_summary.
            Bei Fehler/Timeout: Fallback-Ergebnis (route="chat").
        """
        try:
            return await asyncio.wait_for(
                self._klassifiziere_intern(user_input, letzte_turns or []),
                timeout=_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            log.debug("IntentClassifier Timeout (>%.1fs) → Fallback chat", _TIMEOUT_S)
            return IntentErgebnis()
        except Exception as e:
            log.debug("IntentClassifier Fehler: %s → Fallback chat", e)
            return IntentErgebnis()

    async def _klassifiziere_intern(
        self,
        user_input: str,
        letzte_turns: list[dict],
    ) -> IntentErgebnis:
        """Interner Call — wird von klassifiziere() mit Timeout gewrappt."""
        # Kontext auf max 6 Turns kürzen (3 User + 3 Nova), älteste zuerst
        kontext_turns = letzte_turns[-6:] if len(letzte_turns) > 6 else letzte_turns

        nachrichten = list(kontext_turns)
        nachrichten.append({"role": "user", "content": user_input})

        antwort: str = await self._ollama.chat(
            nachrichten=nachrichten,
            modell=self._modell,
            system=_SYSTEM,
            optionen={"temperature": 0.0, "num_predict": 128},
        )

        return self._parse(antwort)

    def _parse(self, antwort: str) -> IntentErgebnis:
        """Parst JSON aus LLM-Antwort robust — toleriert Markdown-Fences."""
        antwort = antwort.strip()

        # Markdown-Fences entfernen falls LLM trotzdem welche schreibt
        antwort = re.sub(r"```(?:json)?", "", antwort).strip().rstrip("`").strip()

        # JSON-Objekt extrahieren
        m = re.search(r"\{[^{}]*\}", antwort, re.DOTALL)
        if not m:
            log.debug("IntentClassifier: kein JSON in Antwort: %r", antwort[:80])
            return IntentErgebnis()

        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            log.debug("IntentClassifier JSON-Fehler: %s", e)
            return IntentErgebnis()

        route = str(data.get("route", "chat")).lower()
        if route not in ("chat", "agent"):
            route = "chat"

        komplex = bool(data.get("komplex", False))
        agent_summary = data.get("agent_summary") or None
        if isinstance(agent_summary, str):
            agent_summary = agent_summary.strip() or None

        return IntentErgebnis(
            route=route,
            komplex=komplex,
            agent_summary=agent_summary,
        )
