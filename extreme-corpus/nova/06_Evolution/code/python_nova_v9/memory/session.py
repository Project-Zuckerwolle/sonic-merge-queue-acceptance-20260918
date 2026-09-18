"""Nova v9 – Session.

Eine Klasse, eine Aufgabe: den vollstaendigen Gespraechskontext verwalten.

v8-Problem geloest:
    v8 hatte ContextManager (eigene Nachrichten-Liste) UND MemoryCoordinator
    (eigene Session-Liste) die beide unabhaengig wuchsen und beide in den
    LLM-Kontext flossen. Ergebnis: Duplikate, unkontrolliertes Wachstum.

v9-Loesung:
    Session ist die einzige Quelle fuer den Gespraeches-Kontext.
    - Eine Nachrichten-Liste (kein Duplikat)
    - Token-Budget integriert (kein separater ContextManager)
    - Brain-Treffer werden direkt beim Aufbau einbezogen
    - Skill-Ergebnisse werden direkt beim Aufbau einbezogen

Zusammenspiel:
    session = Session(cfg, ollama, brain, episodic, working)
    # Nach jeder Nutzer-Nachricht:
    kontext = session.baue_kontext(user_input, q_vec, skill_ergebnisse)
    messages, system = kontext.als_llm_nachrichten()
    # Nach der Antwort:
    session.speichere(user_input, antwort)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.logger import get
from core.config import get as cget

if TYPE_CHECKING:
    from core.ollama_client   import OllamaClient
    from core.brain_manager   import BrainManager
    from memory.working_memory  import WorkingMemory
    from memory.episodic_memory import EpisodicMemory
    from memory.semantic_memory import SemanticMemory

log = get("session")

# ~4 Zeichen = 1 Token (grobe Schaetzung, ausreichend fuer Budget-Check)
_ZEICHEN_PRO_TOKEN = 4


def _tokens(text: str) -> int:
    return max(1, len(text) // _ZEICHEN_PRO_TOKEN)


# ─── Kontext-Ergebnis ─────────────────────────────────────────────────────────

@dataclass
class Kontext:
    """Ergebnis von session.baue_kontext() — alle Daten fuer den LLM-Aufruf."""
    brain_treffer:    list[dict]        = field(default_factory=list)
    session_eintraege: list[dict]       = field(default_factory=list)
    skill_ergebnisse:  dict[str, str]   = field(default_factory=dict)
    system_prompt:     str              = ""
    token_auslastung:  float            = 0.0
    dauer_ms:          float            = 0.0

    def als_llm_nachrichten(
        self,
        user_input: str,
    ) -> tuple[list[dict], str | None]:
        """
        Baut (messages, system) fuer ollama.stream().
        messages: Session-History + aktuelle Nutzer-Nachricht
        system:   System-Prompt + Brain-Wissen + Skill-Ergebnisse
        """
        system_teile: list[str] = []

        if self.system_prompt:
            system_teile.append(self.system_prompt)

        if self.brain_treffer:
            brain_text = "\n".join(
                f"- {t.get('titel', '?')}: {str(t.get('inhalt', t.get('titel', '')))[:200]}"
                for t in self.brain_treffer[:4]
            )
            system_teile.append(f"Relevantes Wissen:\n{brain_text}")

        if self.skill_ergebnisse:
            skill_text = "\n".join(
                f"[{name}]: {inhalt[:400]}"
                for name, inhalt in self.skill_ergebnisse.items()
            )
            system_teile.append(f"Skill-Ergebnisse:\n{skill_text}")

        system = "\n\n".join(system_teile) if system_teile else None

        messages = [
            {"role": e["rolle"], "content": e["inhalt"]}
            for e in self.session_eintraege
            if e.get("inhalt")
        ]
        messages.append({"role": "user", "content": user_input})

        return messages, system


# ─── Session ──────────────────────────────────────────────────────────────────

class Session:
    """
    Einzige Quelle des Gespraechskontexts.

    Erstellt einmal beim App-Start, bleibt fuer die gesamte Laufzeit aktiv.
    Verwaltet intern: Nachrichten-History, Token-Budget, Memory-Koordination.
    """

    def __init__(
        self,
        cfg:      dict,
        ollama:   "OllamaClient",
        brain:    "BrainManager",
        episodic: "EpisodicMemory",
        working:  "WorkingMemory",
        semantic: "SemanticMemory",
    ) -> None:
        self._ollama   = ollama
        self._brain    = brain
        self._episodic = episodic
        self._working  = working
        self._semantic = semantic

        # Token-Budget
        self._max_tokens    = cget(cfg, "context", "max_tokens",        default=6000)
        self._compress_thr  = cget(cfg, "context", "compress_schwelle", default=0.85)
        self._compress_ziel = cget(cfg, "context", "compress_ziel",     default=0.50)

        # Memory-Einstellungen
        self._max_brain   = cget(cfg, "memory", "max_brain_treffer",       default=4)
        self._max_session = cget(cfg, "memory", "max_kontext_nachrichten",  default=8)

        # System-Prompt (wird von App gesetzt)
        self._system_prompt: str = ""

        # Interne Nachrichten-Liste (eine einzige, kein Duplikat mehr)
        self._verlauf: list[dict] = []   # {role, content, tokens}
        self._komprimierungen: int = 0

        log.info(
            f"Session initialisiert | max_tokens={self._max_tokens} | "
            f"compress_thr={self._compress_thr:.0%}"
        )

    # ─── System-Prompt ────────────────────────────────────────────────────────

    def setze_system(self, text: str) -> None:
        self._system_prompt = text
        log.debug(f"System-Prompt gesetzt | {_tokens(text)} Tokens geschaetzt")

    # ─── Kontext aufbauen ─────────────────────────────────────────────────────

    def baue_kontext(
        self,
        user_input:      str,
        q_vec:           list[float] | None,
        skill_ergebnisse: dict[str, str] | None = None,
    ) -> Kontext:
        """
        Baut den vollstaendigen Kontext fuer einen LLM-Aufruf.
        Einziger Einstiegspunkt — kein anderer Code baut LLM-Kontext.
        """
        t0 = time.monotonic()

        # 1. Brain-Suche (Langzeitgedaechtnis)
        brain_treffer = self._semantic.suche(user_input, top_n=self._max_brain)
        log.debug(f"Brain: {len(brain_treffer)} Treffer")

        # 2. Session-History (Episodisches Gedaechtnis)
        session_eintraege = self._episodic.baue_kontext(q_vec, max_n=self._max_session)
        log.debug(f"Session: {len(session_eintraege)} Eintraege")

        # 3. Working Memory aktualisieren
        self._working.attention_hinzu(user_input[:300], "user", relevanz=1.0)

        # 4. Token-Budget pruefen, ggf. komprimieren
        auslastung = self._token_auslastung()
        if auslastung >= self._compress_thr:
            log.info(f"Token-Budget bei {auslastung:.0%} - komprimiere")
            self._komprimiere()
            auslastung = self._token_auslastung()

        kontext = Kontext(
            brain_treffer     = brain_treffer,
            session_eintraege = session_eintraege,
            skill_ergebnisse  = skill_ergebnisse or {},
            system_prompt     = self._system_prompt,
            token_auslastung  = auslastung,
            dauer_ms          = round((time.monotonic() - t0) * 1000, 1),
        )

        log.debug(
            f"Kontext bereit | brain={len(brain_treffer)} session={len(session_eintraege)} "
            f"skills={len(skill_ergebnisse or {})} budget={auslastung:.0%} | {kontext.dauer_ms}ms"
        )
        return kontext

    # ─── Nach Antwort speichern ───────────────────────────────────────────────

    def speichere(self, user_input: str, antwort: str) -> None:
        """
        Speichert Austausch in allen Memory-Ebenen.
        Wird EINMAL nach jeder Antwort aufgerufen — kein Duplikat.
        """
        # Episodic Memory (Disk-Persistenz)
        self._episodic.hinzufuegen("user",      user_input)
        self._episodic.hinzufuegen("assistant", antwort)

        # Working Memory (RAM, fuer naechste Anfrage)
        self._working.attention_hinzu(antwort[:300], "assistant", relevanz=0.8)

        # Internes Verlauf-Budget (fuer Token-Tracking)
        self._verlauf.append({"role": "user",      "content": user_input, "tokens": _tokens(user_input)})
        self._verlauf.append({"role": "assistant", "content": antwort,    "tokens": _tokens(antwort)})

        log.debug(f"Gespeichert | user={len(user_input)}Z | assistant={len(antwort)}Z | verlauf={len(self._verlauf)} Msgs")

    # ─── Token-Budget ─────────────────────────────────────────────────────────

    def _token_auslastung(self) -> float:
        system_tokens  = _tokens(self._system_prompt)
        verlauf_tokens = sum(m["tokens"] for m in self._verlauf)
        return min(1.0, (system_tokens + verlauf_tokens) / self._max_tokens)

    def _komprimiere(self) -> None:
        """Komprimiert aeltere Nachrichten via LLM zu einer Zusammenfassung."""
        if len(self._verlauf) < 4:
            return

        ziel_tokens = int(self._max_tokens * self._compress_ziel)
        # Behalte die letzten Nachrichten, fasse den Rest zusammen
        behalten_tokens = 0
        behalten = 0
        for msg in reversed(self._verlauf):
            behalten_tokens += msg["tokens"]
            behalten += 1
            if behalten_tokens >= ziel_tokens // 2:
                break

        behalten = max(2, min(behalten, len(self._verlauf) - 2))
        alt       = self._verlauf[:-behalten]
        neu       = self._verlauf[-behalten:]

        if not alt:
            return

        # LLM-Zusammenfassung
        kontext_text = "\n".join(
            f"{m['role'].upper()}: {m['content'][:400]}" for m in alt
        )
        prompt = (
            "Fasse diesen Gespraechsverlauf in 2-3 Saetzen zusammen. "
            "Behalte alle wichtigen Fakten und Themen:\n\n" + kontext_text
        )
        zusammenfassung = self._ollama.generiere_schnell(prompt, max_tokens=300)
        if not zusammenfassung:
            zusammenfassung = f"[{len(alt)} frueheren Nachrichten komprimiert]"

        zusammen_msg = {
            "role":    "system",
            "content": f"[Zusammenfassung]: {zusammenfassung}",
            "tokens":  _tokens(zusammenfassung),
        }
        vorher = len(self._verlauf)
        self._verlauf = [zusammen_msg] + neu
        self._komprimierungen += 1
        log.info(
            f"Komprimiert | {vorher} -> {len(self._verlauf)} Msgs "
            f"| #{self._komprimierungen}"
        )

    # ─── Delegation an Memory-Ebenen ──────────────────────────────────────────

    def brain_suche(self, query: str, top_n: int = 5) -> list[dict]:
        """Direkter Brain-Zugriff (fuer API-Endpunkte)."""
        return self._semantic.suche(query, top_n=top_n)

    def archiviere(self) -> None:
        """Session archivieren und neu starten."""
        self._episodic.archivieren()
        self._working.reset()
        self._verlauf.clear()
        self._komprimierungen = 0
        log.info("Session archiviert und neu gestartet")

    # ─── Status ───────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "token_auslastung_pct": round(self._token_auslastung() * 100, 1),
            "max_tokens":           self._max_tokens,
            "verlauf_msgs":         len(self._verlauf),
            "komprimierungen":      self._komprimierungen,
            "working_attention":    len(self._working.attention_kontext()),
            "episodic":             self._episodic.status(),
            "semantic":             self._semantic.status(),
        }

    def reset(self) -> None:
        """Vollstaendiger Reset (fuer Tests)."""
        self._verlauf.clear()
        self._komprimierungen = 0
        self._working.reset()
