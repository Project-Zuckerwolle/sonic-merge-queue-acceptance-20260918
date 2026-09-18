"""Wetter-Skill: Aktuelles Wetter via wttr.in."""
from core.skill_registry import SkillContext, SkillResult
import requests

def on_message(ctx: SkillContext) -> SkillResult | None:
    query = ctx.skill_config.get("wttr_query", "Berlin")
    # Aus Keywords ggf. Ort überschreiben
    ort_kws = [kw for kw in ctx.keywords if len(kw) > 3 and kw[0].isupper()]
    if ort_kws:
        query = ort_kws[0]
    try:
        r = requests.get(
            f"https://wttr.in/{query}?format=3",
            timeout=8, headers={"User-Agent": "Nova/8.0"}
        )
        if r.status_code == 200:
            return SkillResult(inhalt=f"Wetter für {query}: {r.text.strip()}", typ="info")
        return SkillResult(inhalt=f"Wetter-API nicht erreichbar (HTTP {r.status_code})", typ="info")
    except Exception as e:
        return SkillResult(inhalt=f"Wetter nicht abrufbar: {e}", typ="info")
