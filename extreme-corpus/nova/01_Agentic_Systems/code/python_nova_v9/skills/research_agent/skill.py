"""Research-Agent-Skill: Websuche + Brain + LLM-Synthese."""
from __future__ import annotations
import urllib.parse
import requests
from core.skill_registry import SkillContext, SkillResult

_SYNTHESE_PROMPT = """Synthestisiere die folgenden Recherche-Ergebnisse zu einem klaren, strukturierten Bericht.
Thema: {thema}

Quellen:
{quellen}

Brain-Wissen:
{brain}

Schreibe einen präzisen Bericht auf Deutsch. Maximal 400 Wörter."""


def on_message(ctx: SkillContext) -> SkillResult | None:
    thema = ctx.user_input.strip()
    query = " ".join(ctx.keywords[:6]) if ctx.keywords else thema[:80]

    # 1. Websuche
    web_ergebnisse = _websuche(query)

    # 2. Brain-Wissen
    brain_text = ""
    if ctx.brain_hits:
        brain_text = "\n".join(
            f"- {h.get('titel','?')}: {str(h.get('inhalt', h.get('titel','')))[:200]}"
            for h in ctx.brain_hits[:3]
        )

    # 3. Synthese via LLM (über ollama direkt - wird im ctx.config verfügbar gemacht)
    quellen_text = "\n\n".join(web_ergebnisse[:3]) if web_ergebnisse else "Keine Web-Ergebnisse."

    if not web_ergebnisse and not brain_text:
        return SkillResult(
            inhalt=f"Recherche zu '{thema}': Keine externen Quellen erreichbar. "
                   f"Brain-Einträge: {len(ctx.brain_hits)}.",
            typ="info"
        )

    bericht = (
        f"**Recherche: {thema}**\n\n"
        f"**Web-Ergebnisse:**\n{quellen_text[:1200]}\n\n"
        + (f"**Brain-Wissen:**\n{brain_text}\n\n" if brain_text else "")
        + f"*(Research-Agent: {len(web_ergebnisse)} Quellen, "
          f"{len(ctx.brain_hits)} Brain-Treffer)*"
    )

    return SkillResult(
        inhalt=bericht,
        typ="info",
        metadaten={
            "thema": thema,
            "quellen_anzahl": len(web_ergebnisse),
            "brain_treffer": len(ctx.brain_hits),
            # Chain: Ergebnis ins Brain schreiben
            "chain_skills": ["DateiZugriff"] if ctx.skill_config.get("ins_brain_schreiben") else [],
        }
    )


def _websuche(query: str) -> list[str]:
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            timeout=8, headers={"User-Agent": "Nova/8.0"}
        )
        if r.status_code != 200:
            return []
        data = r.json()
        ergebnisse = []
        if data.get("AbstractText"):
            ergebnisse.append(f"Zusammenfassung: {data['AbstractText'][:500]}")
        if data.get("Answer"):
            ergebnisse.append(f"Direkte Antwort: {data['Answer']}")
        for item in data.get("RelatedTopics", [])[:4]:
            if isinstance(item, dict) and item.get("Text"):
                ergebnisse.append(item["Text"][:300])
        return ergebnisse
    except Exception:
        return []
