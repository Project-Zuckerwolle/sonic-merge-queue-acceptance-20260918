"""Nova Predator v3 — MemoryExtraktion.
LLM-basierte Memory-Extraktion nach jedem Chat-Turn.

Ersetzt den regelbasierten BrainExtractor für den normalen Chat-Pfad.
Läuft als fire-and-forget Task nach jedem Turn in nova_ws._handle_chat().

Vorteile gegenüber dem alten Ansatz:
  - Kein Regex — funktioniert bei Tippfehlern, Kleinschreibung, jeder Formulierung
  - Kein separates Gate-LLM (qwen2.5:3b) — dasselbe Chat-Modell bewertet
  - Das Modell hat den Gesprächskontext bereits im Kopf
  - Einfacher, zuverlässiger, weniger VRAM-Druck

Invarianten (NICHT verändert):
  - BrainManager.add() API unverändert
  - BrainEntry Datenstruktur unverändert
  - BM25Index.add() unverändert
  - BrainIndex wird nicht inkrementell aktualisiert (BrainThinker macht das)
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from core.brain_manager import BrainEntry
from core.event_bus import EventTyp
from core.logger import get

if TYPE_CHECKING:
    from core.brain_index import BM25Index, BrainIndex
    from core.brain_manager import BrainManager
    from core.ollama_client import OllamaClient

log = get("memory_extraktion")

# ── Extraktion-Prompt ─────────────────────────────────────────────────────────

_EXTRAKTION_PROMPT = """\
Du hast gerade dieses Gespräch geführt:

USER: {user_input}

NOVA: {antwort}

Extrahiere daraus maximal 3 Fakten die es wert sind dauerhaft gespeichert zu werden.

Speichere NUR wenn:
- Es ist persönlich (betrifft den User, sein System, seine Projekte)
- Es ist spezifisch und konkret
- Es ist langfristig relevant (in 30 Tagen noch wichtig)

Speichere NICHT:
- Allgemeines Wissen (was jeder weiß)
- Smalltalk oder temporäre Aussagen
- Dinge die du gerade selbst gesagt hast (nur User-Fakten)

Antworte NUR mit einem JSON-Array — kein anderer Text, keine Erklärung:
[
  {{"inhalt": "Fakt in einem Satz", "typ": "fakt", "tags": ["tag1", "tag2"]}},
  ...
]

Gültige Typen: fakt, praeferenz, aufgabe, idee
Wenn nichts speicherungswürdig ist: []
"""


async def extrahiere_und_speichere(
    ollama: "OllamaClient",
    brain_manager: "BrainManager",
    brain_index: "BrainIndex",
    bm25_index: "BM25Index | None",
    modell: str,
    embed_modell: str,
    user_input: str,
    antwort: str,
    conversation_id: str | None = None,
) -> list[str]:
    """Extrahiert Memory-würdige Fakten und speichert sie im Brain.

    Wird als fire-and-forget Task aufgerufen — Fehler werden geloggt,
    nie nach oben propagiert.

    Args:
        ollama:        OllamaClient-Instanz
        brain_manager: BrainManager für Persistenz
        brain_index:   BrainIndex für Vektor-Suche
        bm25_index:    BM25Index für Keyword-Suche (optional)
        modell:        Chat-Modell (bereits warm im VRAM)
        embed_modell:  Embedding-Modell
        user_input:    Letzte User-Nachricht
        antwort:       Nova's Antwort

    Returns:
        Liste der gespeicherten Inhalte (für Logging)
    """
    # Zu kurze Turns überspringen — nichts Relevantes zu merken
    if len(user_input.strip()) < 10 or len(antwort.strip()) < 10:
        return []

    prompt = _EXTRAKTION_PROMPT.format(
        user_input=user_input[:600],
        antwort=antwort[:600],
    )

    try:
        antwort_roh = await ollama.chat(
            nachrichten=[{"role": "user", "content": prompt}],
            modell=modell,
            optionen={
                "temperature": 0.1,
                "num_predict": 600,
                # Thinking-Mode explizit deaktivieren — qwen3 würde sonst
                # <think>...</think> vor das JSON setzen und JSON-Parse bricht.
                # qwen2.5 (nano) hat keinen Thinking-Mode, schadet aber nicht.
                "think": False,
            },
        )
    except Exception as e:
        log.warning("Extraktion LLM-Call fehlgeschlagen: %s", e)
        return []

    # Think-Tags entfernen (Sicherheitsnetz falls think:False ignoriert wird)
    from core.ollama_client import _strip_think_tags
    antwort_roh = _strip_think_tags(antwort_roh)

    # JSON parsen — robust gegen Markdown-Fences und Text-Präambel
    fakten_roh = antwort_roh.strip()
    # Markdown-Fences entfernen
    if fakten_roh.startswith("```"):
        zeilen = fakten_roh.splitlines()
        fakten_roh = "\n".join(
            z for z in zeilen if not z.strip().startswith("```")
        ).strip()
    # JSON-Array aus Text extrahieren falls Präambel davor steht
    if not fakten_roh.startswith("["):
        start = fakten_roh.find("[")
        end   = fakten_roh.rfind("]")
        if start != -1 and end != -1 and end > start:
            fakten_roh = fakten_roh[start:end + 1]
        else:
            log.debug("Extraktion: kein JSON-Array gefunden: %s", fakten_roh[:100])
            return []

    try:
        fakten = json.loads(fakten_roh)
        if not isinstance(fakten, list):
            log.debug("Extraktion: JSON ist keine Liste: %s", type(fakten))
            return []
    except json.JSONDecodeError:
        log.debug("Extraktion: kein valides JSON: %s", fakten_roh[:100])
        return []

    if not fakten:
        log.debug("Extraktion: keine speicherungswürdigen Fakten gefunden")
        return []

    gespeichert: list[str] = []

    for fakt in fakten[:3]:  # Max 3 pro Turn
        inhalt = fakt.get("inhalt", "").strip() if isinstance(fakt, dict) else ""
        if len(inhalt) < 10:
            continue

        typ = fakt.get("typ", "fakt") if isinstance(fakt, dict) else "fakt"
        # Nur gültige Typen zulassen
        if typ not in ("fakt", "praeferenz", "aufgabe", "idee"):
            typ = "fakt"

        tags: list[str] = fakt.get("tags", []) if isinstance(fakt, dict) else []
        if not isinstance(tags, list):
            tags = []

        # Einfacher Duplikat-Check — ohne LLM, nur Textsuche
        try:
            existing = await brain_manager.suche_text(inhalt[:50], max_ergebnisse=3)
            if any(inhalt[:40].lower() in e.inhalt.lower() for e in existing):
                log.debug("Extraktion: Duplikat übersprungen: %s", inhalt[:50])
                continue
        except Exception:
            pass  # Bei Fehler: lieber speichern als überspringen

        # Embedding berechnen (für zukünftige Vektor-Suche)
        vektor: list[float] | None = None
        try:
            v = await ollama.embed(inhalt, modell=embed_modell)
            if v:
                vektor = v
        except Exception:
            pass  # Ohne Vektor trotzdem speichern — BM25 funktioniert noch

        # BrainEntry erstellen und speichern
        entry = BrainEntry(
            id=str(uuid.uuid4()),
            typ=typ,  # type: ignore[arg-type]
            inhalt=inhalt,
            quelle="chat",
            tags=tags,
            vertrauen=0.8,
            vektor=vektor,
            erstellt=datetime.now(timezone.utc).isoformat(),
            zuletzt_bestaetigt=datetime.now(timezone.utc).isoformat(),
            conversation_id=conversation_id,  # v3.1: Herkunft tracken
        )

        try:
            await brain_manager.add(entry)
        except Exception as e:
            log.error("Brain-Speichern fehlgeschlagen: %s", e)
            continue

        # BM25 sofort aktualisieren — für nächste Suche verfügbar
        if bm25_index is not None:
            try:
                bm25_index.add(entry.id, inhalt)
            except Exception:
                pass

        # BrainIndex (Vektor) wird vom BrainThinker periodisch neu aufgebaut
        # Kein inkrementelles add() nötig

        # EventBus-Signal — BrainThinker + alle Subscriber werden informiert
        try:
            from web.state import st as _st
            if hasattr(_st, "bus"):
                await _st.bus.publish(
                    EventTyp.BRAIN_ENTRY_NEU,
                    {"id": entry.id, "typ": typ, "inhalt": inhalt[:60]},
                )
        except Exception:
            pass

        log.info("Memory gespeichert [%s]: %s", typ, inhalt[:60])
        gespeichert.append(inhalt)

    return gespeichert


async def extrahiere_sicher(
    ollama: "OllamaClient",
    brain_manager: "BrainManager",
    brain_index: "BrainIndex",
    bm25_index: "BM25Index | None",
    modell: str,
    embed_modell: str,
    user_input: str,
    antwort: str,
    conversation_id: str | None = None,
) -> None:
    """Fire-and-forget Wrapper — fängt alle Exceptions ab.

    Wird direkt als asyncio.create_task() aufgerufen.
    Fehler werden geloggt aber nie propagiert damit der Chat-Flow
    niemals durch Memory-Extraktion unterbrochen wird.
    """
    try:
        gespeichert = await extrahiere_und_speichere(
            ollama=ollama,
            brain_manager=brain_manager,
            brain_index=brain_index,
            bm25_index=bm25_index,
            modell=modell,
            embed_modell=embed_modell,
            user_input=user_input,
            antwort=antwort,
            conversation_id=conversation_id,
        )
        if gespeichert:
            log.debug("Memory-Extraktion: %d Einträge gespeichert", len(gespeichert))
    except Exception as e:
        log.error("Memory-Extraktion unerwarteter Fehler: %s", e)
