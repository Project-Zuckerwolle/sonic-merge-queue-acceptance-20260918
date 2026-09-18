"""Nova Predator v1 Layer 5 — Task Planner.

Gemma4 zerlegt die Aufgabe in Subtasks und stellt Rückfragen.
Gibt einen strukturierten Plan zurück.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from layer5.agent_task import AgentTask, Subtask, AgentStatus


PLANUNGS_PROMPT = """Du bist ein erfahrener Senior-Entwickler. Du bekommst eine Aufgabe und erstellst SOFORT einen Ausführungsplan — ohne unnötige Rückfragen.

AUFGABE:
{aufgabe}

ENTSCHEIDE SELBST bei unklaren Details:
- Primärsortierung: Dateityp, dann sekundär nach Name/Größe
- Konfiguration: YAML-Datei (einfacher für User)
- Sortierreihenfolge: aufsteigend alphabetisch / Größe aufsteigend
- Pfade: konfigurierbar via Config-Datei
- Konflikte: Zähler-Suffix (datei_1.txt, datei_2.txt)
- UI-Framework: wenn UI gefragt → tkinter (Python stdlib, keine Installation)

Antworte NUR mit einem JSON-Objekt. Kein Text davor oder danach:
{{
  "rückfragen": [],
  "hat_ui": false,
  "subtasks": [
    {{
      "id": "1",
      "beschreibung": "Konkrete Beschreibung, was genau implementiert wird"
    }}
  ],
  "zusammenfassung": "Ein Satz was das Programm macht"
}}

REGELN (strikt):
- "rückfragen" ist IMMER eine leere Liste [] — du entscheidest selbst
- "hat_ui" ist true wenn eine GUI/UI/Oberfläche gebaut wird
- 3-7 Subtasks, jeder konkret und abgeschlossen umsetzbar
- Kein Subtask "Plan erstellen", "Testen", "Dokumentation" — nur Code-Arbeit
- Verwende sinnvolle Defaults wenn der User etwas nicht spezifiziert hat
"""

VERIFIKATIONS_PROMPT = """Der Coding-Agent hat folgenden Subtask ausgeführt:

SUBTASK: {subtask}

ERGEBNIS DES AGENTS:
{ergebnis}

GESAMTPLAN:
{plan}

Bewerte ob der Subtask erfolgreich abgeschlossen wurde.
Antworte NUR mit JSON:
{{
  "erfolgreich": true,
  "weiter": true,
  "korrektur": "",
  "begruendung": "Kurze Begründung"
}}

Falls nicht erfolgreich: beschreibe in "korrektur" genau was fehlt oder falsch ist.
"""


class TaskPlanner:
    """Gemma4 plant Aufgaben und verifiziert Ergebnisse."""

    def __init__(self, ollama_host: str = "http://localhost:11434",
                 modell: str = "gemma4:e4b") -> None:
        self._host   = ollama_host
        self._modell = modell

    async def plane_aufgabe(self, aufgabe: str,
                             kontext: str = "") -> dict:
        """Zerlegt Aufgabe in Subtasks. Gibt Plan-Dict zurück. Stellt NIE Rückfragen."""
        aufgabe_komplett = aufgabe
        if kontext:
            aufgabe_komplett += f"\n\nZusatzinfo vom User:\n{kontext}"

        prompt  = PLANUNGS_PROMPT.format(aufgabe=aufgabe_komplett)
        antwort = await self._llm(prompt, max_tokens=2000)

        plan = self._extrahiere_json(antwort)
        if plan is None:
            plan = {
                "rückfragen":  [],
                "hat_ui":      False,
                "subtasks":    [{"id": "1", "beschreibung": aufgabe}],
                "zusammenfassung": aufgabe,
            }

        # Rückfragen immer ignorieren — Agent entscheidet selbst
        plan["rückfragen"] = []
        return plan

    async def verifiziere_subtask(self, subtask: Subtask,
                                   ergebnis: str, plan_text: str) -> dict:
        """Prüft ob Subtask erfolgreich war."""
        prompt  = VERIFIKATIONS_PROMPT.format(
            subtask=subtask.beschreibung,
            ergebnis=ergebnis[:3000],
            plan=plan_text[:2000],
        )
        antwort = await self._llm(prompt, max_tokens=500)
        result  = self._extrahiere_json(antwort)

        if result is None:
            # Default: erfolgreich wenn kein Fehler-Keyword
            success = not any(w in ergebnis.lower()
                               for w in ["error", "fehler", "exception", "traceback"])
            result = {
                "erfolgreich": success,
                "weiter":      True,
                "korrektur":   "",
                "begruendung": "Automatische Bewertung",
            }
        return result

    def erstelle_task(self, task_id: str, aufgabe: str,
                       projekt: str, workspace, plan: dict) -> AgentTask:
        """Erstellt AgentTask aus Plan-Dict."""
        from pathlib import Path
        task = AgentTask(
            id=task_id, aufgabe=aufgabe, projekt=projekt,
            workspace=Path(workspace),
            plan_text=plan.get("zusammenfassung", aufgabe),
            rückfragen=plan.get("rückfragen", []),
        )
        for s in plan.get("subtasks", []):
            task.subtasks.append(Subtask(
                id=f"{task_id}-{s['id']}",
                beschreibung=s["beschreibung"],
            ))
        return task

    async def _llm(self, prompt: str, max_tokens: int = 1000) -> str:
        """Ruft Gemma4 via Ollama auf."""
        import urllib.request
        import json as json_mod

        url  = f"{self._host}/api/generate"
        body = {
            "model":       self._modell,
            "prompt":      prompt,
            "stream":      False,
            "options":     {"temperature": 0.1, "num_predict": max_tokens},
        }
        try:
            data = json_mod.dumps(body).encode()
            req  = urllib.request.Request(url, data=data,
                                          headers={"Content-Type": "application/json"})
            # asyncio.get_event_loop() ist deprecated ab Python 3.10 und wirft
            # RuntimeError ab 3.12 wenn kein laufender Loop im aktuellen Thread.
            # Da wir bereits in einem async-Kontext sind, nutzen wir run_in_executor.
            loop = asyncio.get_running_loop()
            def _do():
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return json_mod.loads(resp.read().decode())
            result = await loop.run_in_executor(None, _do)
            return result.get("response", "")
        except Exception as e:
            return f'{{"rückfragen":[],"subtasks":[{{"id":"1","beschreibung":"{str(e)[:100]}"}}],"zusammenfassung":"Fehler"}}'

    @staticmethod
    def _extrahiere_json(text: str) -> dict | None:
        """Extrahiert JSON aus LLM-Antwort."""
        # Direkt parsen
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        # ```json ... ``` Block
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # Erstes { ... } im Text
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None
