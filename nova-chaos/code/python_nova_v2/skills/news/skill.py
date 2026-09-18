"""Nova Predator v1 Skill — News.

Liest RSS-Feeds aus konfigurierten Quellen.
Kein API-Key nötig. Kategorien: allgemein, tech, wirtschaft.
"""
from __future__ import annotations

import re
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime

from core.logger import get
from core.skill_registry import SkillContext, SkillResult

log = get("skill.news")


@dataclass
class Artikel:
    titel: str
    link: str
    beschreibung: str
    datum: str
    quelle: str


def _rss_holen(url: str, quelle: str, max_artikel: int) -> list[Artikel]:
    """Lädt RSS-Feed und parst Artikel."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Nova-v10-News/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            inhalt = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return []

    artikel: list[Artikel] = []
    try:
        # Namespace-Probleme abfangen
        inhalt_clean = re.sub(r' xmlns(?::\w+)?="[^"]*"', "", inhalt)
        root = ET.fromstring(inhalt_clean)

        # RSS 2.0 und RDF/RSS 1.0 unterstützen
        items = root.findall(".//item")[:max_artikel]
        for item in items:
            titel = (item.findtext("title") or "").strip()
            link  = (item.findtext("link") or "").strip()
            desc  = (item.findtext("description") or "").strip()
            datum = (item.findtext("pubDate") or item.findtext("date") or "").strip()

            # HTML aus Beschreibung entfernen
            desc = re.sub(r"<[^>]+>", "", desc)[:200].strip()

            if titel:
                artikel.append(Artikel(
                    titel=titel[:120],
                    link=link,
                    beschreibung=desc,
                    datum=datum[:30],
                    quelle=quelle,
                ))
    except ET.ParseError:
        pass

    return artikel


def _kategorie_aus_input(user_input: str) -> str | None:
    """Erkennt gewünschte Kategorie aus User-Input."""
    text = user_input.lower()
    if any(w in text for w in ["tech", "technologie", "software", "computer", "it", "ki", "ai"]):
        return "tech"
    if any(w in text for w in ["wirtschaft", "börse", "aktie", "finanzen", "dax"]):
        return "wirtschaft"
    if any(w in text for w in ["sport", "fußball", "bundesliga"]):
        return "sport"
    return None


def tool_definition() -> dict:
    """Tool-Definition für Gemma Tool-Calling (v3)."""
    return {
        "type": "function",
        "function": {
            "name": "news",
            "description": "Holt aktuelle Nachrichten aus deutschen RSS-Feeds (Tagesschau, Heise, Spiegel, Golem). Nutze dieses Tool bei Fragen nach aktuellen Nachrichten, News oder Schlagzeilen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kategorie": {
                        "type": "string",
                        "description": "Nachrichtenkategorie: 'allgemein' für allgemeine News, 'tech' für Technologie-News",
                        "enum": ["allgemein", "tech"],
                    },
                },
                "required": [],
            },
        },
    }


def on_message(ctx: SkillContext) -> SkillResult | None:
    """Holt Nachrichten aus konfigurierten RSS-Feeds."""
    cfg         = ctx.skill_config
    max_art     = int(cfg.get("max_artikel", 5))
    quellen_cfg = cfg.get("quellen", [])
    std_kats    = cfg.get("standard_kategorien", ["allgemein"])

    # Kategorie aus Input oder Default
    kat_filter = _kategorie_aus_input(ctx.user_input)
    log.info("Skill 'news' aufgerufen: kategorie=%s, max_artikel=%d",
             kat_filter or "auto", max_art)
    if kat_filter is None:
        kat_filter_liste = std_kats
    else:
        kat_filter_liste = [kat_filter]

    # Passende Quellen filtern
    quellen_aktiv = [
        q for q in quellen_cfg
        if q.get("kategorie", "allgemein") in kat_filter_liste
    ]
    if not quellen_aktiv:
        quellen_aktiv = quellen_cfg[:2]  # Fallback: erste zwei

    lines: list[str] = []
    alle_artikel: list[Artikel] = []

    for quelle_cfg in quellen_aktiv:
        name = quelle_cfg.get("name", "?")
        url  = quelle_cfg.get("url", "")
        if not url:
            continue
        artikel = _rss_holen(url, name, max_art)
        alle_artikel.extend(artikel)

    if not alle_artikel:
        # Feeds nicht erreichbar — None zurückgeben damit Nova normal antwortet
        # statt einen sichtbaren Fehler zu zeigen
        return None

    # Ausgabe bauen
    kat_anzeige = kat_filter or "allgemein"
    lines.append(f"**Aktuelle Nachrichten ({kat_anzeige})**\n")

    # Quellen gruppieren
    quellen_map: dict[str, list[Artikel]] = {}
    for art in alle_artikel:
        quellen_map.setdefault(art.quelle, []).append(art)

    for quelle_name, artikel_liste in quellen_map.items():
        lines.append(f"**📰 {quelle_name}**")
        for art in artikel_liste[:max_art]:
            lines.append(f"• **{art.titel}**")
            if art.beschreibung:
                lines.append(f"  {art.beschreibung}")
        lines.append("")

    return SkillResult(
        inhalt="\n".join(lines).strip(),
        typ="daten",
        metadaten={
            "quellen": list(quellen_map.keys()),
            "artikel_gesamt": len(alle_artikel),
            "kategorie": kat_filter,
        },
    )
