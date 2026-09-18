"""Websuche-Skill: DuckDuckGo Instant Answer + HTML-Suche."""
from __future__ import annotations
import re
import urllib.parse
from core.skill_registry import SkillContext, SkillResult
import requests

def on_message(ctx: SkillContext) -> SkillResult | None:
    query = ctx.user_input.strip()
    # Keywords als kompaktere Suchanfrage nutzen
    if ctx.keywords:
        query = " ".join(ctx.keywords[:6])

    try:
        # DuckDuckGo Instant Answer API
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            timeout=8, headers={"User-Agent": "Nova/8.0"}
        )
        if r.status_code == 200:
            data = r.json()
            abstract = data.get("AbstractText", "")
            answer   = data.get("Answer", "")
            related  = [r["Text"] for r in data.get("RelatedTopics", [])[:3]
                        if isinstance(r, dict) and "Text" in r]

            teile = []
            if answer:
                teile.append(f"Direkte Antwort: {answer}")
            if abstract:
                teile.append(f"Zusammenfassung: {abstract[:500]}")
            if related:
                teile.append("Verwandte Themen:\n" + "\n".join(f"- {t[:150]}" for t in related))

            if teile:
                return SkillResult(inhalt="\n\n".join(teile), typ="info",
                                   metadaten={"query": query})

        return SkillResult(
            inhalt=f"Websuche für '{query}' ergab keine Direktantwort. "
                   f"Suche-URL: https://duckduckgo.com/?q={urllib.parse.quote(query)}",
            typ="info"
        )
    except Exception as e:
        return SkillResult(inhalt=f"Websuche fehlgeschlagen: {e}", typ="info")
