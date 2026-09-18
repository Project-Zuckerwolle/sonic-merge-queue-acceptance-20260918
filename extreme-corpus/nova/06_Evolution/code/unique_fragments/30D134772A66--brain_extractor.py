"""Nova Predator v1 — BrainExtractor.
Läuft nach jedem LLM-Output als asyncio.Task (fire-and-forget).
Zerlegt user_input + llm_output in strukturierte Brain-Kandidaten.

Predator Write-Pfad (nach Extraktion, vor brain_manager.add):
  1. Wort-Overlap-Vorfilter   (schnell, kein LLM)
  2. BrainGate                (LLM-Qualitätsbewertung, qwen2.5:3b)
  3. Embedding berechnen      (nur wenn Gate akzeptiert + vollständig)
  4. Vektor-Duplikat-Check    (Cosinus gegen BrainIndex — ersetzt schwachen Text-Check)
  5. TemporalUpdate           (invalidiert veraltete Fakten bei hoher Ähnlichkeit)
  6. brain_manager.add()      (NUR noch gutes Material landet hier)

Python 3.14: asyncio.to_thread() für CPU-gebundene Operationen,
kein asyncio.get_event_loop().
"""
from __future__ import annotations
import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from core.brain_manager import BrainEntry
from core.event_bus import EventBus, EventTyp
from core.logger import get

if TYPE_CHECKING:
    from core.brain_gate import BrainGate
    from core.brain_index import BrainIndex
    from core.brain_manager import BrainManager
    from core.nano_parser import NanoParser
    from core.ollama_client import OllamaClient

log = get("brain_extractor")

# ── Regex-Muster (regelbasierte Extraktion — unverändert) ────────────────────

_PRAEFERENZ = re.compile(
    r"\b(ich mag|ich liebe|ich benutze|ich verwende|ich bevorzuge|"
    r"ich arbeite mit|ich nutze|ich hasse|mir gefällt)\b(.{5,150}?)(?:\.|$)",
    re.IGNORECASE | re.MULTILINE,
)
_FAKT_KURZ = re.compile(
    r"\b([A-Z][a-züäöÜÄÖ\w]{2,40})\s+(ist|sind|heißt|bedeutet|steht für)\s+(.{5,100}?)(?:\.|$)",
    re.MULTILINE,
)
_PYTHON_VERSION = re.compile(r"Python\s+(3\.\d+(?:\.\d+)?)", re.IGNORECASE)
_TECHNOLOGIE = re.compile(
    r"\b(FastAPI|Django|Flask|React|Vue|Docker|Kubernetes|Ollama|Nova|"
    r"PostgreSQL|SQLite|Redis|MongoDB|Windows|Linux|macOS)\b",
    re.IGNORECASE,
)

# ── Temporal Update Prompt ────────────────────────────────────────────────────

_TEMPORAL_PROMPT = """\
Vergleiche diese zwei Fakten:
Alt: "{alt}"
Neu: "{neu}"

Welche Beziehung haben sie?
A = Aktualisierung: Neu ersetzt Alt (z.B. neue Version, neue Adresse, neuer Status)
B = Widerspruch: Beide können nicht gleichzeitig wahr sein
C = Ergänzung: Beide sind unabhängig gültig

Antworte NUR mit JSON: {{"typ": "A"|"B"|"C"}}"""

# Typen für die temporal invalidiert werden können
_TEMPORAL_TYPEN = frozenset({"fakt", "praeferenz"})


@dataclass
class ExtraktionsKandidat:
    typ: str       # 'fakt' | 'praeferenz' | 'aufgabe' | 'idee'
    inhalt: str
    konfidenz: float = 0.7
    tags: list[str] = field(default_factory=list)
    quelle: str = "chat"


class BrainExtractor:
    """Extrahiert Brain-Entries aus Chat-Nachrichten."""

    def __init__(
        self,
        brain_manager: "BrainManager",
        ollama: "OllamaClient",
        nano_parser: "NanoParser",
        bus: EventBus,
        embed_modell: str = "mxbai-embed-large",
        min_score: float = 0.6,
        brain_llm_modell: str = "qwen2.5:3b",
        brain_gate: "BrainGate | None" = None,
        brain_index: "BrainIndex | None" = None,
    ) -> None:
        self._brain            = brain_manager
        self._ollama           = ollama
        self._parser           = nano_parser
        self._bus              = bus
        self._embed_modell     = embed_modell
        self._min_score        = min_score
        self._brain_llm_modell = brain_llm_modell
        self._gate             = brain_gate   # BrainGate — optional, aber empfohlen
        self._index            = brain_index  # BrainIndex — für Vektor-Duplikat-Check

    async def verarbeite(
        self,
        user_input: str,
        llm_output: str,
        q_vec: list[float] | None = None,
    ) -> list[BrainEntry]:
        """Extrahiert und speichert würdige Brain-Entries.

        Args:
            user_input: Nutzereingabe des aktuellen Turns.
            llm_output: Nova's Antwort des aktuellen Turns.
            q_vec:      Embedding-Vektor des user_input (aus Pipeline-Ergebnis).
                        Wird für Vektor-Duplikat-Check genutzt wenn vorhanden.

        Wird nach jedem Chat-Turn aufgerufen (fire-and-forget via TaskManager).
        """
        gespeichert: list[BrainEntry] = []

        # Gate am Anfang des Turns zurücksetzen
        if self._gate:
            self._gate.turn_reset()

        # Stufe 1: Regelbasierte Extraktion (schnell, kein LLM, läuft in Thread)
        kandidaten = await asyncio.to_thread(
            self._regelbasiert_extrahieren,
            user_input,
            llm_output,
        )

        if not kandidaten:
            return []

        # Bekannte Inhalte für Gate-Neuheits-Einschätzung (einmalig laden)
        bekannte_inhalte: list[str] = []
        try:
            alle = await self._brain.alle()
            bekannte_inhalte = [e.inhalt for e in alle[:20]]  # Top 20 für Kontext
        except Exception:
            pass

        # Stufe 2–6: Write-Pipeline für jeden Kandidaten
        for kandidat in kandidaten:
            entry = await self._verarbeite_kandidat(
                kandidat       = kandidat,
                bekannte_inhalte = bekannte_inhalte,
                q_vec          = q_vec,
            )
            if entry:
                gespeichert.append(entry)
                bekannte_inhalte.append(entry.inhalt)  # Aktualisieren für nachfolgende Kandidaten

        if gespeichert:
            log.info("BrainExtractor: %d neue Entries gespeichert", len(gespeichert))

        return gespeichert

    async def _verarbeite_kandidat(
        self,
        kandidat: ExtraktionsKandidat,
        bekannte_inhalte: list[str],
        q_vec: list[float] | None,
    ) -> BrainEntry | None:
        """Führt den vollständigen Write-Pfad für einen Kandidaten durch.

        Returns:
            Den gespeicherten BrainEntry oder None wenn abgelehnt.
        """
        inhalt = kandidat.inhalt.strip()

        # ── Schritt 1: Wort-Overlap-Vorfilter (kein LLM, strict 0.7) ──────
        # ── FuzzyDedup-Vorfilter v2 (Levenshtein-Ratio, threshold 85) ──────
        # Erkennt Tippfehler, Groß/Klein-Varianten, Wortumstellungen
        from core.fuzzy_dedup import ist_duplikat as fuzzy_ist_duplikat
        vorhandene = await self._brain.suche_text(inhalt[:50], max_ergebnisse=5)
        vorhandene_inhalte = [e.inhalt for e in vorhandene]
        dup, match, dup_score = fuzzy_ist_duplikat(inhalt, vorhandene_inhalte, threshold=85)
        if dup:
            log.debug(
                "FuzzyDedup: '%s' ähnlich zu '%s' [score=%.0f]",
                inhalt[:40], match[:40], dup_score,
            )
            return None

        # ── Schritt 2: BrainGate (LLM-Qualitätsbewertung) ──────────────────
        gate_score = 0.0
        gewicht    = "vollstaendig"

        if self._gate:
            gate_result = await self._gate.bewerte(
                fakt             = inhalt,
                quelle           = kandidat.quelle,
                bekannte_inhalte = bekannte_inhalte[:5],
            )
            if not gate_result.akzeptiert:
                log.debug(
                    "Gate ablehnt [%.1f] '%s': %s",
                    gate_result.score, inhalt[:40], gate_result.ablehnungsgrund
                )
                return None
            gate_score = gate_result.score
            gewicht    = gate_result.gewicht
        else:
            # Kein Gate konfiguriert — Mindestlänge als Notfall-Filter
            if len(inhalt) < 15:
                return None

        # ── Schritt 3: Embedding (nur bei vollständigem Gewicht) ────────────
        vektor: list[float] = []
        if gewicht == "vollstaendig":
            try:
                vektor = await self._ollama.embed(inhalt, modell=self._embed_modell)
            except Exception as e:
                log.debug("Embed fehlgeschlagen (fahre ohne Vektor fort): %s", e)
                gewicht = "notiz"  # Kein Embedding → Notiz

        # ── Schritt 4: Vektor-Duplikat-Check (ersetzt schwachen Text-Check) ─
        if vektor and self._index:
            duplikat_treffer = self._index.suche(vektor, max_ergebnisse=2, min_score=0.88)
            if duplikat_treffer:
                # Prüfe ob es tatsächlich ein Duplikat ist (nicht der Entry selbst)
                best_id, best_score = duplikat_treffer[0]
                log.debug(
                    "Vektor-Duplikat [sim=%.3f]: '%s' ≈ '%s'",
                    best_score, inhalt[:30],
                    (await self._brain.get(best_id) or type('', (), {'inhalt': '?'})()).inhalt[:30]
                )
                return None

        # ── Schritt 5: Temporal Update (invalidiert veraltete Fakten) ───────
        if vektor and kandidat.typ in _TEMPORAL_TYPEN:
            await self._temporal_update(inhalt, vektor, kandidat.typ)

        # ── Schritt 6: brain_manager.add() ──────────────────────────────────
        entry = BrainEntry(
            id         = str(uuid.uuid4()),
            typ        = kandidat.typ,  # type: ignore[arg-type]
            inhalt     = inhalt,
            quelle     = kandidat.quelle,
            tags       = kandidat.tags,
            vertrauen  = kandidat.konfidenz,
            vektor     = vektor or None,
            gate_score = gate_score,
            gewicht    = gewicht,
        )
        await self._brain.add(entry)
        await self._bus.publish(EventTyp.BRAIN_ENTRY_NEU, {"id": entry.id, "typ": entry.typ})
        log.info(
            "Brain ✓ [%s|%.1f|%s]: %s",
            entry.typ, gate_score, gewicht, inhalt[:60]
        )
        return entry

    async def _temporal_update(
        self,
        neuer_inhalt: str,
        neuer_vektor: list[float],
        typ: str,
    ) -> None:
        """Invalidiert veraltete Fakten wenn ein neuer sie ersetzt oder widerspricht.

        Läuft nur für Typen in _TEMPORAL_TYPEN (fakt, praeferenz).
        Bei hoher Ähnlichkeit (>0.92) wird das LLM gefragt ob es sich um
        eine Aktualisierung oder einen Widerspruch handelt.
        """
        if not self._index:
            return

        treffer = self._index.suche(neuer_vektor, max_ergebnisse=3, min_score=0.92)
        for entry_id, sim in treffer:
            alter_entry = await self._brain.get(entry_id)
            if not alter_entry or alter_entry.typ not in _TEMPORAL_TYPEN:
                continue
            if getattr(alter_entry, "veraltet", False):
                continue  # Bereits veraltet

            try:
                prompt = _TEMPORAL_PROMPT.format(
                    alt=alter_entry.inhalt[:150],
                    neu=neuer_inhalt[:150],
                )
                antwort = await self._ollama.chat(
                    nachrichten=[{"role": "user", "content": prompt}],
                    modell=self._brain_llm_modell,
                    optionen={"temperature": 0.0, "num_predict": 30},
                )
                # JSON aus Antwort extrahieren
                start = antwort.find("{")
                end   = antwort.rfind("}")
                if start != -1 and end != -1:
                    daten = json.loads(antwort[start:end + 1])
                    if daten.get("typ") in ("A", "B"):
                        alter_entry.veraltet = True
                        await self._brain.vertrauen_anpassen(entry_id, delta=-0.4)
                        log.info(
                            "Temporal [%s|sim=%.2f]: '%s' → veraltet durch '%s'",
                            daten.get("typ"), sim,
                            alter_entry.inhalt[:40], neuer_inhalt[:40]
                        )
            except Exception as e:
                log.debug("TemporalUpdate fehlgeschlagen (kein Block): %s", e)
                # Temporal-Fehler niemals den Write blockieren

    def _regelbasiert_extrahieren(
        self,
        user_input: str,
        llm_output: str,
    ) -> list[ExtraktionsKandidat]:
        """Regelbasierte Extraktion — läuft in asyncio.to_thread().
        Unverändert gegenüber Originalcode — dient als erster Kandidaten-Filter.
        """
        kandidaten: list[ExtraktionsKandidat] = []
        kombiniert = user_input + "\n" + llm_output

        # Präferenzen aus User-Input
        for m in _PRAEFERENZ.finditer(user_input):
            text = (m.group(1) + " " + m.group(2)).strip()
            if len(text) > 10:
                kandidaten.append(ExtraktionsKandidat(
                    typ="praeferenz",
                    inhalt=text[:200],
                    konfidenz=0.85,
                    tags=self._extrahiere_tags(text),
                ))

        # Fakten aus kombiniertem Text
        for m in _FAKT_KURZ.finditer(kombiniert):
            subjekt   = m.group(1)
            praedikat = m.group(2)
            objekt    = m.group(3).strip()
            if len(objekt) > 3 and subjekt[0].isupper():
                inhalt = f"{subjekt} {praedikat} {objekt}"
                kandidaten.append(ExtraktionsKandidat(
                    typ="fakt",
                    inhalt=inhalt[:200],
                    konfidenz=0.75,
                    tags=self._extrahiere_tags(inhalt),
                ))

        # Python-Versionserkennung
        for m in _PYTHON_VERSION.finditer(kombiniert):
            inhalt = f"User nutzt Python {m.group(1)}"
            kandidaten.append(ExtraktionsKandidat(
                typ="fakt",
                inhalt=inhalt,
                konfidenz=0.90,
                tags=["python", "entwicklung"],
            ))

        # Todo-Signal
        parse_r = self._parser.parse(user_input)
        if parse_r.todo_signal and parse_r.todo_text:
            kandidaten.append(ExtraktionsKandidat(
                typ="aufgabe",
                inhalt=parse_r.todo_text[:200],
                konfidenz=0.95,
                tags=["aufgabe"],
            ))

        # Deduplizierung innerhalb der Kandidaten
        einzigartig: list[ExtraktionsKandidat] = []
        gesehen: set[str] = set()
        for k in kandidaten:
            key = k.inhalt[:60].lower()
            if key not in gesehen:
                gesehen.add(key)
                einzigartig.append(k)

        return einzigartig[:10]   # Max 10 Kandidaten pro Turn

    def _extrahiere_tags(self, text: str) -> list[str]:
        """Extrahiert Technologie-Tags aus Text."""
        tags = []
        for m in _TECHNOLOGIE.finditer(text):
            tags.append(m.group(1).lower())
        return list(set(tags))[:5]

    def _ist_duplikat(self, neu: str, vorhanden: str, schwelle: float = 0.70) -> bool:
        """Wort-Overlap-Prüfung (Vorfilter vor Gate und Vektor-Check).

        Schwellenwert auf 0.70 gesenkt gegenüber Original (war 0.85) —
        strengerer Vorfilter, da Gate und Vektor-Check die Hauptlast tragen.
        """
        neu_w = set(neu.lower().split())
        vor_w = set(vorhanden.lower().split())
        if not neu_w or not vor_w:
            return False
        overlap = len(neu_w & vor_w) / min(len(neu_w), len(vor_w))
        return overlap >= schwelle
