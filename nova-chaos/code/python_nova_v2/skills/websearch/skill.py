"""Nova Predator v3.2 Skill — WebSearch.

Verbesserte Websuche:
  - Primär: SearXNG lokal (http://127.0.0.1:8888) — JSON-API,
    aggregiert Google+Bing+DDG+Qwant+Brave.
  - Fallback: DuckDuckGo HTML wenn SearXNG nicht erreichbar.
  - Query aus Tool-Call (Gemma plant sauber) statt rohem user_input.
  - Top-3 Seiten parallel fetchen, Hauptinhalt readability-artig
    extrahieren (<main>/<article>, längste <p>-Kette, kein Boilerplate).
  - Produkt-Heuristik: Geizhals/Idealo/Alternate bei Shopping-Queries.

SearXNG-Setup: searxng-compose.yml im nova-v2/ Root.
"""
from __future__ import annotations

import asyncio
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html import unescape

from core.logger import get
from core.skill_registry import SkillContext, SkillResult

log = get("skill.websearch")


@dataclass
class Suchergebnis:
    titel: str
    url: str
    snippet: str
    score: float = 0.0


_SEARXNG_URL       = "http://127.0.0.1:8888/search"
_SEARXNG_TIMEOUT_S = 8
_DDG_URL           = "https://html.duckduckgo.com/html/"
_DDG_TIMEOUT_S     = 8

_PRODUKT_KEYWORDS = (
    "kaufen", "preis", "vergleich", "test", "review",
    "mainboard", "motherboard", "grafikkarte", "gpu", "cpu", "ram",
    "ssd", "monitor", "laptop", "notebook", "mouse", "tastatur",
    "headset", "smartphone", "handy",
)
_PRODUKT_DOMAINS = (
    "geizhals.de", "idealo.de", "alternate.de", "mindfactory.de",
    "caseking.de", "notebooksbilliger.de", "computerbase.de",
    "pcgameshardware.de", "heise.de",
)


def _ist_produktsuche(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _PRODUKT_KEYWORDS)


# ── Query-Bestimmung ──────────────────────────────────────────────────────

def _query_bestimmen(ctx: SkillContext) -> str:
    """Priorität: skill_config['query'] > keywords > bereinigter user_input."""
    q = str(ctx.skill_config.get("query", "")).strip()
    if q:
        return q
    if ctx.keywords:
        return " ".join(str(k) for k in ctx.keywords).strip()
    return _bereinige_userinput(ctx.user_input)


def _bereinige_userinput(user_input: str) -> str:
    """Rohen User-Text auf Such-tauglichen Kern reduzieren."""
    text = user_input.strip()
    trigger = [
        r"^(?:hey\s+(?:nova\s*,?\s*)?)?(?:suche(?:\s+(?:mal|bitte|nach|online))?|"
        r"google\s+mal|recherchiere(?:\s+(?:f.r\s+mich|mal|bitte|und))?|"
        r"finde(?:\s+(?:mir|bitte))?)\s+",
        r"^(?:was\s+ist|wer\s+ist|wie\s+(?:funktioniert|geht)|was\s+sind)\s+",
    ]
    for pat in trigger:
        text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()

    stoppwoerter = {
        "ein","eine","einen","einem","einer","der","die","das","den","dem",
        "gutes","gute","guten","relativ","ziemlich","ganz",
        "neues","neue","neuen","aktuelles","fähiges","fähige","fähigen",
        "bitte","mal","doch","und","oder",
        "mit","für","von","zu","zur","zum","mir","dir","uns",
    }
    tokens = [t for t in re.split(r"\s+", text) if t.strip()]
    gefiltert = [t for t in tokens if t.lower().strip(".,!?") not in stoppwoerter]
    if len(gefiltert) < 3:
        return text.rstrip("?!.").strip() or user_input.strip()
    return " ".join(gefiltert).rstrip("?!.").strip()


# ── Backend: SearXNG ──────────────────────────────────────────────────────

def _searxng_suche(query: str, max_ergebnisse: int, sprache: str) -> list[Suchergebnis]:
    params = {
        "q": query,
        "format": "json",
        "language": sprache if sprache != "auto" else "de",
        "categories": "general",
        "safesearch": "0",
    }
    url = f"{_SEARXNG_URL}?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Nova-Predator/3.2",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=_SEARXNG_TIMEOUT_S) as resp:
            daten = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        log.debug("SearXNG nicht erreichbar: %s", e)
        return []
    except Exception as e:
        log.warning("SearXNG Fehler: %s", e)
        return []

    rohe = daten.get("results", [])
    if not rohe:
        return []

    ergebnisse = [
        Suchergebnis(
            titel=str(r.get("title", ""))[:150].strip(),
            url=str(r.get("url", "")).strip(),
            snippet=str(r.get("content", ""))[:400].strip(),
            score=float(r.get("score", 0.0)),
        )
        for r in rohe if r.get("url") and r.get("title")
    ]

    if _ist_produktsuche(query):
        def _bonus(e: Suchergebnis) -> float:
            return e.score + (0.5 if any(d in e.url for d in _PRODUKT_DOMAINS) else 0)
        ergebnisse.sort(key=_bonus, reverse=True)

    return ergebnisse[:max_ergebnisse]


# ── Backend: DuckDuckGo (Fallback) ────────────────────────────────────────

def _ddg_suche(query: str, max_ergebnisse: int) -> list[Suchergebnis]:
    q_enc = urllib.parse.quote(query)
    url = f"{_DDG_URL}?q={q_enc}&kl=de-de"
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
        })
        with urllib.request.urlopen(req, timeout=_DDG_TIMEOUT_S) as resp:
            html = resp.read(512 * 1024).decode("utf-8", errors="replace")
    except Exception as e:
        log.warning("DDG-Fallback fehlgeschlagen: %s", e)
        return []

    ergebnisse: list[Suchergebnis] = []
    blocks = re.findall(
        r'<div class="result(?:__body)?"[^>]*>(.*?)(?:</div>\s*</div>)',
        html, flags=re.DOTALL,
    )
    if not blocks:
        return _ddg_legacy(html, max_ergebnisse)

    for block in blocks[:max_ergebnisse * 2]:
        t = re.search(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.+?)</a>',
            block, flags=re.DOTALL,
        )
        if not t:
            continue
        url_raw = t.group(1)
        titel = re.sub(r"<[^>]+>", "", t.group(2)).strip()
        if "uddg=" in url_raw:
            m = re.search(r"uddg=([^&]+)", url_raw)
            real_url = urllib.parse.unquote(m.group(1)) if m else url_raw
        else:
            real_url = url_raw
        s = re.search(r'class="result__snippet"[^>]*>(.+?)</a>', block, flags=re.DOTALL)
        snippet = re.sub(r"<[^>]+>", "", s.group(1)).strip()[:400] if s else ""
        ergebnisse.append(Suchergebnis(
            titel=unescape(titel)[:150], url=real_url, snippet=unescape(snippet),
        ))
        if len(ergebnisse) >= max_ergebnisse:
            break
    return ergebnisse


def _ddg_legacy(html: str, max_ergebnisse: int) -> list[Suchergebnis]:
    titel_pat = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)<', html)
    snip_pat  = re.findall(r'class="result__snippet"[^>]*>(.+?)</a>', html, flags=re.DOTALL)
    snippets  = [re.sub(r"<[^>]+>", "", s).strip() for s in snip_pat]
    out = []
    for i, (url_raw, titel) in enumerate(titel_pat[:max_ergebnisse]):
        if "uddg=" in url_raw:
            m = re.search(r"uddg=([^&]+)", url_raw)
            real_url = urllib.parse.unquote(m.group(1)) if m else url_raw
        else:
            real_url = url_raw
        out.append(Suchergebnis(
            titel=unescape(titel.strip())[:150],
            url=real_url,
            snippet=unescape(snippets[i][:400] if i < len(snippets) else ""),
        ))
    return out


# ── Seiten-Inhalt (readability-artig) ────────────────────────────────────

_BOILERPLATE = re.compile(
    r"<(?:script|style|nav|header|footer|aside|form|iframe|noscript)\b[^>]*>.*?"
    r"</(?:script|style|nav|header|footer|aside|form|iframe|noscript)>",
    flags=re.DOTALL | re.IGNORECASE,
)
_COOKIE = re.compile(
    r"<[^>]*class=\"[^\"]*(?:cookie|banner|consent|gdpr|privacy)[^\"]*\"[^>]*>.*?</\w+>",
    flags=re.DOTALL | re.IGNORECASE,
)


def _seite_inhalt(url: str, max_zeichen: int = 1500) -> str:
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
        })
        with urllib.request.urlopen(req, timeout=6) as resp:
            ct = resp.headers.get("Content-Type", "")
            if "text/html" not in ct and "application/xhtml" not in ct:
                return ""
            html = resp.read(500 * 1024).decode("utf-8", errors="replace")
    except Exception as e:
        log.debug("Seiten-Fetch %s: %s", url[:60], e)
        return ""

    html = _BOILERPLATE.sub(" ", html)
    html = _COOKIE.sub(" ", html)

    main_m = re.search(
        r"<(?:main|article)\b[^>]*>(.*?)</(?:main|article)>",
        html, flags=re.DOTALL | re.IGNORECASE,
    )
    if main_m:
        kern = main_m.group(1)
    else:
        paras = re.findall(r"<p\b[^>]*>(.*?)</p>", html, flags=re.DOTALL | re.IGNORECASE)
        sinnvoll = [p for p in paras if len(re.sub(r"<[^>]+>", "", p)) > 80]
        kern = " ".join(sinnvoll) if sinnvoll else html

    text = re.sub(r"<[^>]+>", " ", kern)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_zeichen]


async def _seiten_parallel(urls: list[str], max_zeichen: int) -> list[str]:
    if not urls:
        return []
    results = await asyncio.gather(
        *(asyncio.to_thread(_seite_inhalt, u, max_zeichen) for u in urls),
        return_exceptions=True,
    )
    return [r if isinstance(r, str) else "" for r in results]


# ── Tool-Definition ───────────────────────────────────────────────────────

def tool_definition() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "websearch",
            "description": (
                "Sucht aktuelle Informationen im Web. Nutze bei aktuellen "
                "Ereignissen, Produktempfehlungen, unbekannten Fakten oder "
                "expliziten Recherche-Wünschen. Formuliere die Query kurz "
                "und präzise (3-6 Wörter), ohne Füllwörter. Beispiel: "
                "User fragt 'Welches DDR5-Mainboard mit 3 GPU-Slots AM5?' "
                "→ query='AMD X670E Mainboard DDR5 3 PCIe Slots'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Präziser Suchbegriff, 3-6 Wörter, keine Füllwörter.",
                    },
                },
                "required": ["query"],
            },
        },
    }


# ── Haupt-Hook ────────────────────────────────────────────────────────────

async def on_message(ctx: SkillContext) -> SkillResult | None:
    cfg          = ctx.skill_config
    max_erg      = int(cfg.get("max_ergebnisse", 8))
    fetch_seiten = int(cfg.get("fetch_seiten", 3))
    max_zeichen  = int(cfg.get("max_zeichen_pro_seite", 1500))
    sprache      = str(cfg.get("sprache", "de"))

    query = _query_bestimmen(ctx)
    if not query:
        return SkillResult(inhalt="🔍 Kein Suchbegriff erkannt.", typ="warnung")

    if len(query.split()) > 10:
        log.warning("Query sehr lang (%d Wörter) — Gemma hat evtl. nicht geplant: %r",
                    len(query.split()), query)

    log.info("Websuche: query=%r, max=%d, seiten=%d", query, max_erg, fetch_seiten)

    ergebnisse = await asyncio.to_thread(_searxng_suche, query, max_erg, sprache)
    quelle = "SearXNG"

    if not ergebnisse:
        log.info("SearXNG leer — DDG-Fallback")
        ergebnisse = await asyncio.to_thread(_ddg_suche, query, max_erg)
        quelle = "DuckDuckGo (Fallback)"

    if not ergebnisse:
        return SkillResult(
            inhalt=(
                f"🔍 Keine Suchergebnisse für: *{query}*\n\n"
                f"Prüfe ob SearXNG läuft: "
                f"`curl http://127.0.0.1:8888/search?q=test&format=json`"
            ),
            typ="warnung",
        )

    seiten_urls    = [e.url for e in ergebnisse[:fetch_seiten]]
    seiten_inhalte = await _seiten_parallel(seiten_urls, max_zeichen)

    lines: list[str] = [f"**🔍 Websuche: {query}**  _(Quelle: {quelle})_\n"]
    for i, erg in enumerate(ergebnisse):
        lines.append(f"**{i+1}. {erg.titel}**")
        if erg.snippet:
            lines.append(f"   {erg.snippet}")
        lines.append(f"   🔗 {erg.url}")
        lines.append("")

    for i, (erg, inhalt) in enumerate(zip(ergebnisse[:fetch_seiten], seiten_inhalte)):
        if inhalt:
            lines.append(f"---\n**Inhalt Treffer {i+1} ({erg.url[:70]}):**")
            lines.append(inhalt)
            lines.append("")

    return SkillResult(
        inhalt="\n".join(lines).strip(),
        typ="daten",
        metadaten={
            "query":         query,
            "treffer":       len(ergebnisse),
            "seiten_geholt": sum(1 for s in seiten_inhalte if s),
            "quelle":        quelle,
        },
    )
