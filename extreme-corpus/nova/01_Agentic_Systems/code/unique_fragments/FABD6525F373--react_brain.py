"""Nova Predator v1 Layer 6 — ReAct-Brain.

Der Kern-Loop: Denken → Handeln → Beobachten → Reflektieren → [wiederholen].

Unterschied zu Layer 5 (Coding-Agent):
  - Layer 5: sequenzieller Plan, linearer Ablauf, Code-fokussiert
  - Layer 6: echter ReAct-Loop, kann Plan verwerfen und neu planen,
             General-Purpose, Research + Computer-Steuerung + Skills

Modelle:
  - Gemma4 (gemma4:e4b):  Planen, Reflektieren, Memory-Kompaktierung
  - Codestral (codestral:22b): Tool-Calls ausführen (falls verfügbar)
  - Fallback: Gemma4 für alles wenn Codestral nicht verfügbar
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, TYPE_CHECKING

from core.logger import get
from layer6.apex_types import (
    ApexTask, ApexStatus, ApexMemory, ReActStep, ReActTyp, TraceLog
)

if TYPE_CHECKING:
    from core.ollama_client import OllamaClient

log = get("apex.brain")

# ── System-Prompts ────────────────────────────────────────────────────────────

REACT_SYSTEM = """Du bist Nova APEX — ein autonomer Reasoning-Agent.

Du arbeitest in einem ReAct-Loop: Denken → Handeln → Beobachten → Wiederholen.

DEINE AUFGABE:
{aufgabe}

AKTUELLER KONTEXT:
{kontext}

VERFÜGBARE TOOLS:
{tools}

AUSGABE-FORMAT (strikt einhalten):
Antworte immer mit einem JSON-Objekt:

{{
  "thought": "Dein Gedankengang — was weißt du, was brauchst du, warum tust du das?",
  "action": "tool_name",
  "action_input": {{...}}
}}

ODER wenn du fertig bist:
{{
  "thought": "Die Aufgabe ist abgeschlossen weil...",
  "action": "FINISH",
  "action_input": {{"ergebnis": "Vollständige Zusammenfassung des Ergebnisses"}}
}}

ODER wenn du den Plan neu aufstellen musst:
{{
  "thought": "Mein bisheriger Ansatz funktioniert nicht weil... Ich ändere die Strategie:",
  "action": "REPLAN",
  "action_input": {{"neuer_plan": ["Schritt 1", "Schritt 2", ...]}}
}}

REGELN:
- Denke laut nach bevor du handelst
- Nach 3 fehlgeschlagenen Actions: REPLAN
- Nach REPLAN: frischer Ansatz, kein stures Wiederholen
- PFLICHT: Speichere alle Ergebnisse mit datei_schreiben BEVOR du FINISH aufrufst:
    Code → .py/.js/.html Datei, Dokumente → .md Datei, Daten → .json/.yaml
- FINISH erst wenn Datei(en) geschrieben sind und Aufgabe wirklich erledigt ist
- Schreibe im Ergebnis-Text: welche Datei du erstellt hast
- Antworte NUR mit JSON, kein Text davor oder danach
"""

REFLEXIONS_SYSTEM = """Du bist ein kritischer Analyst. Bewerte den bisherigen Fortschritt.

AUFGABE: {aufgabe}
BISHERIGE ARBEIT: {verlauf}

Antworte NUR mit JSON:
{{
  "fortschritt_prozent": 0-100,
  "was_funktioniert": "...",
  "was_fehlt": "...",
  "strategie_anpassen": true/false,
  "naechster_fokus": "...",
  "extrahierte_fakten": ["Fakt 1", "Fakt 2"]
}}
"""

KOMPAKTIERUNGS_SYSTEM = """Fasse den bisherigen Arbeitsverlauf kompakt zusammen.
Behalte alle wichtigen Fakten, Ergebnisse und Entscheidungen.
Verwirf unwichtige Details.

VERLAUF:
{verlauf}

Antworte mit einer kompakten Zusammenfassung (max 300 Wörter):
"""


class ReActBrain:
    """Der ReAct-Kern-Loop für Layer 6.

    Wird vom ApexOrchestrator aufgerufen.
    Gibt Events per AsyncGenerator zurück.
    """

    # Tool-Registry: Name → (beschreibung, handler_name)
    TOOLS: dict[str, dict] = {
        "websearch": {
            "beschreibung": "Sucht im Web nach aktuellen Informationen. Input: {query: str}",
            "requires": "websearch_skill",
        },
        "webfetch": {
            "beschreibung": "Lädt Inhalt einer URL. Input: {url: str}",
            "requires": None,
        },
        "skill_ausfuehren": {
            "beschreibung": "Führt einen Nova-Skill aus (wetter, news, finanzen). Input: {skill: str, user_input: str}",
            "requires": "skill_mesh",
        },
        "datei_schreiben": {
            "beschreibung": "Schreibt Text in eine Datei im Workspace. Input: {dateiname: str, inhalt: str}",
            "requires": None,
        },
        "datei_lesen": {
            "beschreibung": "Liest eine Datei aus dem Workspace. Input: {dateiname: str}",
            "requires": None,
        },
        "notiz": {
            "beschreibung": "Speichert eine wichtige Notiz im Memory. Input: {text: str}",
            "requires": None,
        },
        "code_ausfuehren": {
            "beschreibung": "Führt Python-Code aus. Input: {code: str}",
            "requires": None,
        },
        "brain_suche": {
            "beschreibung": "Sucht in Novas Langzeit-Memory (Brain). Input: {query: str}",
            "requires": "brain_manager",
        },
        # ── Layer-5-Tools (Coding-Agent-Fähigkeiten) ─────────────────────────
        "bash": {
            "beschreibung": "Führt Shell-Befehl im Workspace aus (Windows CMD/PowerShell). Input: {command: str, timeout?: int}",
            "requires": None,
        },
        "edit_file": {
            "beschreibung": "Ersetzt Text in einer bestehenden Datei. Input: {dateiname: str, old_string: str, new_string: str}",
            "requires": None,
        },
        "glob_search": {
            "beschreibung": "Findet Dateien per Muster im Workspace. Input: {pattern: str}",
            "requires": None,
        },
        "grep_search": {
            "beschreibung": "Durchsucht Datei-Inhalte per Regex. Input: {pattern: str, glob?: str}",
            "requires": None,
        },
        "design_ui": {
            "beschreibung": "Generiert vollständige UI-Datei (tkinter/html). Input: {framework: str, beschreibung: str, ausgabe_datei: str, komponenten?: list}",
            "requires": None,
        },
        "powershell": {
            "beschreibung": "Führt PowerShell-Befehl aus (nur Windows). Input: {command: str}",
            "requires": None,
        },
        "briefing": {
            "beschreibung": "Generiert Daily Briefing aus Wetter, News, Finanzen und Todos. Input: {} (keine Args nötig)",
            "requires": "briefing",
        },
    }

    # Coding-Tools die codestral:22b statt Gemma4 nutzen
    CODING_TOOLS = frozenset({
        "bash", "edit_file", "datei_schreiben", "datei_lesen",
        "glob_search", "grep_search", "code_ausfuehren",
        "design_ui", "powershell",
    })

    def __init__(
        self,
        ollama: "OllamaClient",
        chat_modell:  str = "gemma4:e4b",
        coder_modell: str = "codestral:22b",
        max_iter: int = 40,
        reflexion_intervall: int = 8,
    ) -> None:
        self._ollama        = ollama
        self._chat_modell   = chat_modell
        self._coder_modell  = coder_modell
        self._max_iter      = max_iter
        self._reflexion_intervall = reflexion_intervall

    async def run(
        self,
        task: ApexTask,
        extra_tools: dict[str, Any] | None = None,
        hat_code: bool = False,
    ) -> AsyncGenerator[dict, None]:
        """Hauptloop — führt Task aus, yieldet Events.

        hat_code=True: Tool-Calls mit codestral:22b ausführen (schneller, besser für Code).
        Planung/Reflexion bleibt immer bei Gemma4.
        """
        task.status = ApexStatus.RUNNING
        trace = TraceLog(task.workspace, task.id)
        tools = {**self.TOOLS, **(extra_tools or {})}
        tools_beschreibung = self._tools_als_text(tools)
        # Aktives Modell für Tool-Calls
        self._aktives_modell = self._coder_modell if hat_code else self._chat_modell
        if hat_code:
            log.debug("Apex nutzt codestral:22b für Coding-Tasks")

        yield {"typ": "react_start", "task_id": task.id, "aufgabe": task.aufgabe}

        for iteration in range(1, self._max_iter + 1):
            task.iteration = iteration
            yield {"typ": "iteration", "n": iteration, "max": self._max_iter}

            # ── Kontext-Kompaktierung wenn nötig ──────────────────────
            if task.memory.braucht_kompaktierung():
                yield {"typ": "memory_flush", "grund": "Scratchpad voll"}
                zusammenfassung = await self._kompaktiere(task)
                task.memory.kompaktiere(zusammenfassung)
                schritt = ReActStep(
                    typ=ReActTyp.MEMORY_FLUSH,
                    inhalt=f"Kompaktiert: {zusammenfassung[:100]}"
                )
                task.log_schritt(schritt)
                trace.schreibe(schritt, task.id)

            # ── Stagnations-Check ──────────────────────────────────────
            if task.fortschritt_stagniert(6):
                yield {"typ": "warnung", "text": "Fortschritt stagniert — erzwinge Reflexion"}
                replan_schritt = await self._erzwinge_replan(task)
                task.log_schritt(replan_schritt)
                trace.schreibe(replan_schritt, task.id)
                yield {"typ": "replan", "neuer_plan": task.plan}
                continue

            # ── LLM-Call: Denken + Handeln ────────────────────────────
            kontext = task.memory.als_kontext()
            system = REACT_SYSTEM.format(
                aufgabe=task.aufgabe,
                kontext=kontext or "Kein bisheriger Kontext — dies ist der erste Schritt.",
                tools=tools_beschreibung,
            )

            t0 = time.monotonic()
            llm_antwort = await self._llm_call(system, task)
            dauer_ms = int((time.monotonic() - t0) * 1000)

            if llm_antwort is None:
                yield {"typ": "fehler", "text": "LLM nicht erreichbar"}
                task.status = ApexStatus.FAILED
                return

            # ── Parse: Thought + Action ───────────────────────────────
            parsed = self._parse_react(llm_antwort)
            if parsed is None:
                # LLM hat kein valides JSON geliefert → Retry-Gedanke
                task.memory.add_thought(f"[Parse-Fehler] Antwort war kein valides JSON: {llm_antwort[:100]}")
                continue

            thought   = parsed.get("thought", "")
            action    = parsed.get("action", "")
            act_input = parsed.get("action_input", {})

            # Thought-Schritt loggen
            thought_schritt = ReActStep(
                typ=ReActTyp.THOUGHT,
                inhalt=thought,
                dauer_ms=dauer_ms,
            )
            task.memory.add_thought(thought)
            task.log_schritt(thought_schritt)
            trace.schreibe(thought_schritt, task.id)
            yield {"typ": "thought", "text": thought[:300]}

            # ── FINISH? ───────────────────────────────────────────────
            if action == "FINISH":
                ergebnis = act_input.get("ergebnis", thought)
                task.ergebnis = ergebnis
                task.status   = ApexStatus.DONE
                task.done_at  = time.time()

                # Ergebnis automatisch als output.md speichern falls noch keine Datei da
                ausgabe_pfad = task.workspace / "output.md"
                if not ausgabe_pfad.exists() and ergebnis.strip():
                    try:
                        task.workspace.mkdir(parents=True, exist_ok=True)
                        ausgabe_pfad.write_text(ergebnis, encoding="utf-8")
                        log.info("Apex: Ergebnis gespeichert → %s", ausgabe_pfad)
                    except Exception as e:
                        log.warning("Apex: output.md konnte nicht gespeichert werden: %s", e)

                fertig_schritt = ReActStep(typ=ReActTyp.OBSERVATION,
                                           inhalt=f"FERTIG: {ergebnis[:200]}", success=True)
                task.log_schritt(fertig_schritt)
                trace.schreibe(fertig_schritt, task.id)
                yield {
                    "typ": "fertig",
                    "ergebnis": ergebnis,
                    "zusammenfassung": task.zusammenfassung(),
                    "output_datei": "output.md" if ausgabe_pfad.exists() else "",
                }
                return

            # ── REPLAN? ───────────────────────────────────────────────
            if action == "REPLAN":
                neuer_plan = act_input.get("neuer_plan", [])
                task.plan = neuer_plan
                task.status = ApexStatus.REPLANNING
                replan_schritt = ReActStep(
                    typ=ReActTyp.REPLAN,
                    inhalt=f"Neuer Plan: {json.dumps(neuer_plan, ensure_ascii=False)[:200]}"
                )
                task.log_schritt(replan_schritt)
                trace.schreibe(replan_schritt, task.id)
                yield {"typ": "replan", "neuer_plan": neuer_plan}
                task.status = ApexStatus.RUNNING
                continue

            # ── ACTION ausführen ──────────────────────────────────────
            action_schritt = ReActStep(
                typ=ReActTyp.ACTION,
                tool=action,
                tool_args=act_input,
            )
            yield {"typ": "action", "tool": action, "args": act_input}

            ergebnis_text, ok = await self._tool_ausfuehren(
                action, act_input, task, extra_tools
            )

            action_schritt.tool_result = ergebnis_text[:800]
            action_schritt.success = ok
            task.log_schritt(action_schritt)
            trace.schreibe(action_schritt, task.id)

            # Observation
            obs_schritt = ReActStep(
                typ=ReActTyp.OBSERVATION,
                inhalt=ergebnis_text[:600],
                success=ok,
            )
            task.memory.add_observation(ergebnis_text, limit=obs_laenge)
            task.log_schritt(obs_schritt)
            trace.schreibe(obs_schritt, task.id)
            yield {
                "typ":     "observation",
                "tool":    action,
                "success": ok,
                "text":    ergebnis_text[:400],
            }

            # ── Reflexion alle N Iterationen ──────────────────────────
            if iteration % self._reflexion_intervall == 0:
                yield {"typ": "reflexion_start"}
                reflexion = await self._reflektiere(task)
                if reflexion:
                    for fakt in reflexion.get("extrahierte_fakten", []):
                        task.memory.add_fact(fakt)
                        task.brain_eintraege.append(fakt)  # → Brain eintragen
                    yield {
                        "typ": "reflexion",
                        "fortschritt": reflexion.get("fortschritt_prozent", 0),
                        "naechster_fokus": reflexion.get("naechster_fokus", ""),
                        "fakten": reflexion.get("extrahierte_fakten", []),
                    }

        # Max-Iterationen erreicht
        task.status  = ApexStatus.FAILED
        task.fehler  = f"Max. Iterationen ({self._max_iter}) erreicht ohne FINISH"
        task.done_at = time.time()
        yield {"typ": "max_iter", "text": task.fehler}

    # ── Private Hilfsmethoden ─────────────────────────────────────────────────

    async def _llm_call(self, system: str, task: ApexTask,
                         fuer_aktion: str = "") -> str | None:
        """LLM-Call: Gemma4 für Planung/Reflexion, codestral für Coding-Tool-Calls."""
        # Modell-Wahl: Coding-Tools → codestral, alles andere → Gemma4
        modell = self._chat_modell
        if fuer_aktion and fuer_aktion in self.CODING_TOOLS:
            if hasattr(self, "_aktives_modell"):
                modell = self._aktives_modell
        try:
            nachrichten: list[dict] = []
            relevante = [s for s in task.schritte[-6:] if s.inhalt]
            for s in relevante:
                rolle = "assistant" if s.typ in (ReActTyp.THOUGHT, ReActTyp.REPLAN) else "user"
                nachrichten.append({"role": rolle, "content": s.inhalt[:300]})

            if not nachrichten:
                nachrichten = [{"role": "user", "content": task.aufgabe}]

            antwort = await self._ollama.chat(
                nachrichten=nachrichten,
                modell=modell,
                system=system,
                optionen={"temperature": 0.15, "num_predict": 1200, "format": "json"},
            )
            return antwort
        except Exception as e:
            log.error("ReAct LLM-Fehler (modell=%s): %s", modell, e)
            return None

    def _parse_react(self, text: str) -> dict | None:
        """Extrahiert JSON aus LLM-Antwort."""
        # Direkt
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        # ```json ... ```
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # Erstes { ... }
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None

    async def _tool_ausfuehren(
        self,
        tool_name: str,
        args: dict,
        task: ApexTask,
        extra_tools: dict | None,
    ) -> tuple[str, bool]:
        """Führt ein Tool aus. Gibt (ergebnis, success) zurück."""
        # Extra-Handler vom Orchestrator (z.B. Computer-Controller)
        if extra_tools and tool_name in extra_tools:
            handler = extra_tools[tool_name]
            try:
                ergebnis = await handler(args) if asyncio.iscoroutinefunction(handler) else handler(args)
                return str(ergebnis)[:1000], True
            except Exception as e:
                return f"Fehler: {e}", False

        # Eingebaute Tools
        if tool_name == "websearch":
            return await self._tool_websearch(args)
        elif tool_name == "webfetch":
            return await self._tool_webfetch(args)
        elif tool_name == "datei_schreiben":
            return self._tool_datei_schreiben(args, task.workspace)
        elif tool_name == "datei_lesen":
            return self._tool_datei_lesen(args, task.workspace)
        elif tool_name == "notiz":
            text = args.get("text", "")
            task.memory.add_fact(text)
            return f"Notiz gespeichert: {text[:100]}", True
        elif tool_name == "code_ausfuehren":
            return await self._tool_code(args, task.workspace)
        elif tool_name == "brain_suche":
            return await self._tool_brain_suche(args)
        elif tool_name == "skill_ausfuehren":
            return await self._tool_skill(args)
        # ── Layer-5-Erweiterungen ──────────────────────────────────
        elif tool_name == "bash":
            return await self._tool_bash(args, task.workspace)
        elif tool_name == "edit_file":
            return self._tool_edit_file(args, task.workspace)
        elif tool_name == "glob_search":
            return self._tool_glob_search(args, task.workspace)
        elif tool_name == "grep_search":
            return self._tool_grep_search(args, task.workspace)
        elif tool_name == "design_ui":
            return self._tool_design_ui(args, task.workspace)
        elif tool_name == "powershell":
            return await self._tool_powershell(args, task.workspace)
        elif tool_name == "briefing":
            return await self._tool_briefing()
        else:
            return f"Unbekanntes Tool: {tool_name}", False

    async def _tool_websearch(self, args: dict) -> tuple[str, bool]:
        """DuckDuckGo-Suche."""
        import urllib.request
        import urllib.parse
        query = args.get("query", "")
        if not query:
            return "Kein Suchbegriff angegeben", False
        try:
            q_enc = urllib.parse.quote(query)
            url   = f"https://html.duckduckgo.com/html/?q={q_enc}&kl=de-de"
            req   = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 Nova-APEX/1.0",
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read(256 * 1024).decode("utf-8", errors="replace")
            import re
            hits = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)<', html)
            snips = re.findall(r'class="result__snippet"[^>]*>(.+?)</a>', html)
            snips_clean = [re.sub(r"<[^>]+>", "", s).strip() for s in snips]
            results = []
            for i, (u, t) in enumerate(hits[:6]):
                snippet = snips_clean[i] if i < len(snips_clean) else ""
                results.append(f"**{t.strip()}**\n{snippet[:200]}\n🔗 {u}")
            return "\n\n".join(results) if results else "Keine Ergebnisse", bool(results)
        except Exception as e:
            return f"Suche fehlgeschlagen: {e}", False

    async def _tool_webfetch(self, args: dict) -> tuple[str, bool]:
        """Lädt URL-Inhalt."""
        import urllib.request
        import re
        url = args.get("url", "")
        if not url:
            return "Keine URL angegeben", False
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 Nova-APEX/1.0"
            })
            with urllib.request.urlopen(req, timeout=8) as resp:
                html = resp.read(200 * 1024).decode("utf-8", errors="replace")
            html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
            html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:2000], True
        except Exception as e:
            return f"Fetch fehlgeschlagen: {e}", False

    def _tool_datei_schreiben(self, args: dict, workspace: Path) -> tuple[str, bool]:
        dateiname = args.get("dateiname", "output.txt")
        inhalt    = args.get("inhalt", "")
        # Sicherheit: kein path traversal
        datei = workspace / Path(dateiname).name
        workspace.mkdir(parents=True, exist_ok=True)
        try:
            datei.write_text(inhalt, encoding="utf-8")
            return f"Datei geschrieben: {datei.name} ({len(inhalt)} Zeichen)", True
        except Exception as e:
            return f"Schreib-Fehler: {e}", False

    def _tool_datei_lesen(self, args: dict, workspace: Path) -> tuple[str, bool]:
        dateiname = args.get("dateiname", "")
        datei = workspace / Path(dateiname).name
        if not datei.exists():
            return f"Datei nicht gefunden: {dateiname}", False
        try:
            inhalt = datei.read_text(encoding="utf-8", errors="replace")
            return inhalt[:3000], True
        except Exception as e:
            return f"Lese-Fehler: {e}", False

    async def _tool_code(self, args: dict, workspace: Path) -> tuple[str, bool]:
        """Führt Python-Code aus."""
        import subprocess, sys, tempfile
        code = args.get("code", "")
        if not code.strip():
            return "Kein Code angegeben", False
        workspace.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                         dir=str(workspace), delete=False,
                                         encoding="utf-8") as f:
            f.write(code)
            tmp = f.name
        try:
            proc = subprocess.run(
                [sys.executable, tmp],
                capture_output=True, text=True, timeout=20, cwd=str(workspace), creationflags=_NO_WINDOW
            )
            output = (proc.stdout + proc.stderr).strip()
            return output[:1000] or "(kein Output)", proc.returncode == 0
        except subprocess.TimeoutExpired:
            return "Timeout nach 20s", False
        except Exception as e:
            return f"Ausführungs-Fehler: {e}", False
        finally:
            Path(tmp).unlink(missing_ok=True)

    async def _tool_brain_suche(self, args: dict) -> tuple[str, bool]:
        """Sucht in Novas Brain."""
        query = args.get("query", "")
        try:
            from web.state import st
            if not hasattr(st, "brain_manager"):
                return "Brain nicht verfügbar", False
            treffer = await st.brain_manager.suche_text(query, max_ergebnisse=5)
            if not treffer:
                return f"Keine Brain-Einträge für: {query}", False
            lines = [f"• {e.inhalt[:200]}" for e in treffer]
            return "\n".join(lines), True
        except Exception as e:
            return f"Brain-Suche Fehler: {e}", False

    async def _tool_skill(self, args: dict) -> tuple[str, bool]:
        """Führt einen Nova-Skill aus."""
        skill_name  = args.get("skill", "")
        user_input  = args.get("user_input", "")
        try:
            from web.state import st
            from core.skill_registry import SkillContext, Hook
            if not hasattr(st, "registry"):
                return "Skill-Registry nicht verfügbar", False
            regs = st.registry.get_skill(skill_name)
            on_msg = [r for r in regs if r.hook == Hook.ON_MESSAGE and r.aktiv]
            if not on_msg:
                return f"Skill '{skill_name}' nicht gefunden oder inaktiv", False
            ctx = SkillContext(
                user_input=user_input,
                keywords=tuple(user_input.lower().split()[:5]),
                intents=(), brain_hits=(), session_facts=(),
                skill_config=on_msg[0].config, config={},
            )
            import inspect
            fn = on_msg[0].funktion
            if inspect.iscoroutinefunction(fn):
                result = await fn(ctx)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, fn, ctx)
            if result:
                return result.inhalt, result.typ != "fehler"
            return "Skill lieferte kein Ergebnis", False
        except Exception as e:
            return f"Skill-Fehler: {e}", False

    # ── Layer-5-Tool-Implementierungen ───────────────────────────────────────

    async def _tool_bash(self, args: dict, workspace: "Path") -> tuple[str, bool]:
        """Shell-Befehl im Workspace ausführen."""
        import subprocess
        command = args.get("command", "")
        timeout = int(args.get("timeout", 30))
        if not command.strip():
            return "Kein Befehl angegeben", False
        workspace.mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=str(workspace), creationflags=_NO_WINDOW
            )
            output = (proc.stdout + proc.stderr).strip()
            ok = proc.returncode == 0
            return (output[:2000] or "(kein Output)"), ok
        except subprocess.TimeoutExpired:
            return f"Timeout nach {timeout}s", False
        except Exception as e:
            return f"Bash-Fehler: {e}", False

    def _tool_edit_file(self, args: dict, workspace: "Path") -> tuple[str, bool]:
        """Ersetzt Text in einer Datei."""
        dateiname  = args.get("dateiname", "")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        datei = workspace / Path(dateiname).name
        if not datei.exists():
            return f"Datei nicht gefunden: {dateiname}", False
        if not old_string:
            return "old_string fehlt", False
        try:
            inhalt = datei.read_text(encoding="utf-8", errors="replace")
            if old_string not in inhalt:
                return f"old_string nicht in {dateiname} gefunden", False
            aktualisiert = inhalt.replace(old_string, new_string, 1)
            datei.write_text(aktualisiert, encoding="utf-8")
            return f"✓ {dateiname}: Text ersetzt ({len(old_string)} → {len(new_string)} Zeichen)", True
        except Exception as e:
            return f"Edit-Fehler: {e}", False

    def _tool_glob_search(self, args: dict, workspace: "Path") -> tuple[str, bool]:
        """Findet Dateien per Glob-Muster im Workspace."""
        import fnmatch
        pattern = args.get("pattern", "*")
        treffer = []
        try:
            for p in sorted(workspace.rglob("*")):
                if p.is_file() and fnmatch.fnmatch(p.name, pattern.split("/")[-1]):
                    try:
                        rel = p.relative_to(workspace)
                        treffer.append(str(rel))
                    except ValueError:
                        pass
            if not treffer:
                return f"Keine Dateien für Muster '{pattern}'", False
            return "\n".join(treffer[:50]), True
        except Exception as e:
            return f"Glob-Fehler: {e}", False

    def _tool_grep_search(self, args: dict, workspace: "Path") -> tuple[str, bool]:
        """Sucht Text in Workspace-Dateien."""
        import re, fnmatch
        pattern  = args.get("pattern", "")
        glob_pat = args.get("glob", "*")
        if not pattern:
            return "Kein Such-Pattern angegeben", False
        try:
            regex  = re.compile(pattern, re.IGNORECASE)
            treffer = []
            for datei in sorted(workspace.rglob("*")):
                if not datei.is_file():
                    continue
                if not fnmatch.fnmatch(datei.name, glob_pat):
                    continue
                try:
                    text  = datei.read_text(encoding="utf-8", errors="replace")
                    zeilen = text.splitlines()
                    for i, z in enumerate(zeilen):
                        if regex.search(z):
                            rel = datei.relative_to(workspace)
                            treffer.append(f"{rel}:{i+1}: {z.strip()[:120]}")
                except Exception:
                    continue
            if not treffer:
                return f"Kein Treffer für '{pattern}'", False
            return "\n".join(treffer[:30]), True
        except re.error as e:
            return f"Ungültiges Regex: {e}", False

    def _tool_design_ui(self, args: dict, workspace: "Path") -> tuple[str, bool]:
        """Generiert UI-Datei via Layer-5-Design-Tool."""
        try:
            from layer6.design_tool import generate as ui_generate
            framework    = args.get("framework", "tkinter")
            beschreibung = args.get("beschreibung", "App")
            ausgabe      = args.get("ausgabe_datei", "app.py")
            stil         = args.get("stil", "modern-dark")
            komponenten  = args.get("komponenten", [])

            # Wenn keine Komponenten angegeben: sinnvolle Defaults aus Beschreibung ableiten
            if not komponenten:
                desc_lower = beschreibung.lower()
                komponenten = []
                # Chat/Roleplay Apps
                if any(w in desc_lower for w in ["chat", "roleplay", "nachricht", "konversation"]):
                    komponenten = [
                        "text:Chatverlauf",
                        "entry:Nachricht eingeben",
                        "button:Senden",
                        "button:Verlauf löschen",
                        "label:Modell",
                        "combobox:Modell",
                    ]
                # Datei/Sortierer
                elif any(w in desc_lower for w in ["datei", "sortier", "ordner"]):
                    komponenten = [
                        "entry:Quellordner",
                        "button:Durchsuchen",
                        "button:Sortieren",
                        "listbox:Dateien",
                        "progressbar",
                        "label:Status",
                    ]
                # Allgemein: Basis-Set
                else:
                    komponenten = [
                        "button:Starten",
                        "text:Ausgabe",
                        "label:Status",
                    ]

            code = ui_generate(framework, beschreibung, stil, komponenten)
            workspace.mkdir(parents=True, exist_ok=True)
            datei = workspace / Path(ausgabe).name
            datei.write_text(code, encoding="utf-8")
            zeilen = code.count("\n") + 1
            return f"✓ UI-Datei erstellt: {datei.name} ({zeilen} Zeilen, {framework}, {len(komponenten)} Widgets)", True
        except Exception as e:
            return f"Design-UI Fehler: {e}", False

    async def _tool_powershell(self, args: dict, workspace: "Path") -> tuple[str, bool]:
        """PowerShell-Befehl (nur Windows)."""
        import subprocess, platform
        if platform.system() != "Windows":
            return "PowerShell nur auf Windows verfügbar", False
        command = args.get("command", "")
        if not command.strip():
            return "Kein Befehl angegeben", False
        try:
            shell = "pwsh" if __import__("shutil").which("pwsh") else "powershell"
            proc  = subprocess.run(
                [shell, "-NoProfile", "-NonInteractive", "-Command", command],
                capture_output=True, text=True, timeout=30, cwd=str(workspace), creationflags=_NO_WINDOW
            )
            output = (proc.stdout + proc.stderr).strip()
            return (output[:2000] or "(kein Output)"), proc.returncode == 0
        except Exception as e:
            return f"PowerShell-Fehler: {e}", False

    async def _tool_briefing(self) -> tuple[str, bool]:
        """Generiert Daily Briefing."""
        try:
            from web.state import st
            if not hasattr(st, "briefing"):
                return "Briefing-Service nicht verfügbar", False
            result = await st.briefing.generiere()
            return str(result)[:1500] if result else "Kein Briefing generiert", bool(result)
        except Exception as e:
            return f"Briefing-Fehler: {e}", False

    async def _reflektiere(self, task: ApexTask) -> dict | None:
        """Gemma4 reflektiert den bisherigen Verlauf."""
        verlauf = "\n".join(
            f"[{s.typ.value}] {s.inhalt[:150]}" for s in task.schritte[-15:]
        )
        system = REFLEXIONS_SYSTEM.format(
            aufgabe=task.aufgabe, verlauf=verlauf
        )
        try:
            antwort = await self._ollama.chat(
                nachrichten=[{"role": "user", "content": "Reflektiere jetzt."}],
                modell=self._chat_modell,
                system=system,
                optionen={"temperature": 0.1, "num_predict": 600},
            )
            return self._parse_react(antwort)
        except Exception as e:
            log.warning("Reflexion fehlgeschlagen: %s", e)
            return None

    async def _kompaktiere(self, task: ApexTask) -> str:
        """Gemma4 kompaktiert den Scratchpad."""
        verlauf = "\n".join(task.memory.scratchpad[-20:])
        system  = KOMPAKTIERUNGS_SYSTEM.format(verlauf=verlauf)
        try:
            return await self._ollama.chat(
                nachrichten=[{"role": "user", "content": "Fasse zusammen."}],
                modell=self._chat_modell,
                system=system,
                optionen={"temperature": 0.1, "num_predict": 400},
            )
        except Exception:
            return f"[Kompaktiert nach {task.iteration} Iterationen]"

    async def _erzwinge_replan(self, task: ApexTask) -> ReActStep:
        """Erzwingt einen Replan wenn Fortschritt stagniert."""
        system = f"""Der Agent steckt fest. Erstelle einen komplett neuen Plan.

AUFGABE: {task.aufgabe}
PROBLEM: 3+ aufeinanderfolgende Actions sind fehlgeschlagen.
BISHERIGE FAKTEN: {chr(10).join(task.memory.facts[:5])}

Antworte NUR mit JSON:
{{"neuer_plan": ["Schritt 1", "Schritt 2", "Schritt 3"]}}
"""
        try:
            antwort = await self._ollama.chat(
                nachrichten=[{"role": "user", "content": "Neuer Plan bitte."}],
                modell=self._chat_modell,
                system=system,
                optionen={"temperature": 0.2, "num_predict": 300},
            )
            parsed = self._parse_react(antwort)
            if parsed and "neuer_plan" in parsed:
                task.plan = parsed["neuer_plan"]
        except Exception:
            task.plan = ["Grundlegende Recherche neu starten", "Anderem Ansatz folgen"]

        return ReActStep(
            typ=ReActTyp.REPLAN,
            inhalt=f"Erzwungener Replan: {task.plan[:2]}"
        )

    def _tools_als_text(self, tools: dict) -> str:
        """Formatiert Tool-Liste für System-Prompt."""
        lines = []
        for name, info in tools.items():
            if isinstance(info, dict):
                desc = info.get("beschreibung", "")
            else:
                desc = str(info)
            lines.append(f"- **{name}**: {desc}")
        return "\n".join(lines)
