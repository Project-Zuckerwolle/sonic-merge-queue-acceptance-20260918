"""Nova Predator v1 Skill — Wetter.

Nutzt Open-Meteo API (kostenlos, kein API-Key nötig) + Geocoding.
Erkennt Stadt aus User-Input, sonst Standard-Stadt aus Config.
"""
from __future__ import annotations

import json
import re
import urllib.request
import urllib.parse
from datetime import datetime

from core.logger import get
from core.skill_registry import SkillContext, SkillResult

log = get("skill.wetter")


# Wettercodes → lesbare Beschreibung + Emoji
WETTER_CODES: dict[int, tuple[str, str]] = {
    0:  ("Klar", "☀️"),
    1:  ("Überwiegend klar", "🌤️"),
    2:  ("Teils bewölkt", "⛅"),
    3:  ("Bedeckt", "☁️"),
    45: ("Nebel", "🌫️"),
    48: ("Reifnebel", "🌫️"),
    51: ("Leichter Nieselregen", "🌦️"),
    53: ("Nieselregen", "🌦️"),
    55: ("Starker Nieselregen", "🌧️"),
    61: ("Leichter Regen", "🌧️"),
    63: ("Regen", "🌧️"),
    65: ("Starker Regen", "🌧️"),
    71: ("Leichter Schnee", "🌨️"),
    73: ("Schnee", "❄️"),
    75: ("Starker Schnee", "❄️"),
    80: ("Leichte Schauer", "🌦️"),
    81: ("Schauer", "🌧️"),
    82: ("Starke Schauer", "⛈️"),
    85: ("Schneeschauer", "🌨️"),
    86: ("Starke Schneeschauer", "❄️"),
    95: ("Gewitter", "⛈️"),
    96: ("Gewitter mit Hagel", "⛈️"),
    99: ("Starkes Gewitter", "⛈️"),
}

WOCHENTAGE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


def _geocode(stadt: str) -> tuple[float, float, str] | None:
    """Gibt (lat, lon, display_name) zurück oder None."""
    q = urllib.parse.quote(stadt)
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count=1&language=de"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        results = data.get("results", [])
        if not results:
            return None
        r = results[0]
        name = r.get("name", stadt)
        land  = r.get("country", "")
        return float(r["latitude"]), float(r["longitude"]), f"{name}, {land}"
    except Exception:
        return None


def _wetter_holen(lat: float, lon: float, tage: int) -> dict | None:
    """Ruft Open-Meteo API auf."""
    params = (
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,apparent_temperature,precipitation,weathercode,windspeed_10m"
        f"&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum,sunrise,sunset"
        f"&forecast_days={min(tage, 7)}"
        f"&wind_speed_unit=kmh"
        f"&timezone=auto"
    )
    url = f"https://api.open-meteo.com/v1/forecast?{params}"
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _stadtname_aus_input(user_input: str) -> str | None:
    """Versucht Stadtname aus User-Input zu extrahieren."""
    patterns = [
        r"wetter (?:in|für|bei) ([A-ZÄÖÜa-zäöü\s\-]+?)(?:\?|$|,|\s+(?:morgen|heute|diese))",
        r"in ([A-ZÄÖÜa-zäöü\s\-]{3,20}) (?:wetter|regen|temperatur|warm|kalt)",
        r"([A-Z][a-zäöü]+(?:\s[A-Z][a-zäöü]+)?) weather",
        r"(?:forecast|vorhersage) (?:für|for) ([A-ZÄÖÜa-zäöü\s\-]+?)(?:\?|$)",
    ]
    for pat in patterns:
        m = re.search(pat, user_input, re.IGNORECASE)
        if m:
            kandidat = m.group(1).strip()
            if 2 < len(kandidat) < 30:
                return kandidat
    return None


def tool_definition() -> dict:
    """Tool-Definition für Gemma Tool-Calling (v3)."""
    return {
        "type": "function",
        "function": {
            "name": "wetter",
            "description": "Aktuelles Wetter und Vorhersage für eine Stadt. Nutze dieses Tool bei jeder Frage nach Wetter, Temperatur, Regen, Wind oder Wettervorhersage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stadt": {
                        "type": "string",
                        "description": "Stadtname, z.B. 'Berlin', 'München', 'Wien'",
                    },
                    "tage": {
                        "type": "integer",
                        "description": "Vorhersage-Tage (1-7). Standard: 3",
                    },
                },
                "required": ["stadt"],
            },
        },
    }


def on_message(ctx: SkillContext) -> SkillResult | None:
    """Haupt-Handler: Wetter abrufen und formatiert zurückgeben."""
    cfg  = ctx.skill_config
    tage = int(cfg.get("tage", 3))

    # Stadt aus Input oder Config
    stadt_input = _stadtname_aus_input(ctx.user_input)
    stadt_name  = stadt_input or cfg.get("standard_stadt", "Berlin")
    log.info("Skill 'wetter' aufgerufen: stadt=%s, tage=%d", stadt_name, tage)

    # Geocoding
    geo = _geocode(stadt_name)
    if geo is None:
        return SkillResult(
            inhalt=f"❌ Stadt '{stadt_name}' nicht gefunden.",
            typ="fehler",
        )
    lat, lon, display = geo

    # Wetterdaten
    data = _wetter_holen(lat, lon, tage)
    if data is None:
        return SkillResult(
            inhalt="❌ Wetterdaten konnten nicht abgerufen werden (Open-Meteo offline?).",
            typ="fehler",
        )

    lines: list[str] = []

    # ── Aktuell ─────────────────────────────────────────────────────
    cur = data.get("current", {})
    if cur:
        code = cur.get("weathercode", 0)
        desc, emoji = WETTER_CODES.get(code, ("Unbekannt", "🌡️"))
        temp      = cur.get("temperature_2m", "?")
        feels     = cur.get("apparent_temperature", "?")
        wind      = cur.get("windspeed_10m", "?")
        precip    = cur.get("precipitation", 0)

        lines.append(f"**Wetter in {display}**")
        lines.append(f"{emoji} {desc} · {temp}°C (gefühlt {feels}°C)")
        lines.append(f"💨 Wind {wind} km/h · 🌧️ Niederschlag {precip} mm")
        lines.append("")

    # ── Tagesvorhersage ──────────────────────────────────────────────
    daily = data.get("daily", {})
    dates     = daily.get("time", [])
    codes     = daily.get("weathercode", [])
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    precips   = daily.get("precipitation_sum", [])
    sunrises  = daily.get("sunrise", [])
    sunsets   = daily.get("sunset", [])

    if dates:
        lines.append("**Vorhersage:**")
        for i in range(min(tage, len(dates))):
            try:
                dt  = datetime.fromisoformat(dates[i])
                tag = "Heute" if i == 0 else ("Morgen" if i == 1 else WOCHENTAGE[dt.weekday()])
                desc2, emoji2 = WETTER_CODES.get(codes[i] if i < len(codes) else 0, ("?", "?"))
                tmax  = f"{max_temps[i]:.0f}" if i < len(max_temps) else "?"
                tmin  = f"{min_temps[i]:.0f}" if i < len(min_temps) else "?"
                regen = f"{precips[i]:.1f}" if i < len(precips) else "0"
                lines.append(f"  {tag}: {emoji2} {desc2} · {tmin}–{tmax}°C · {regen}mm")
            except (IndexError, ValueError):
                pass

    # ── Sonnenauf/-untergang ─────────────────────────────────────────
    if sunrises and sunsets:
        try:
            sr = datetime.fromisoformat(sunrises[0]).strftime("%H:%M")
            ss = datetime.fromisoformat(sunsets[0]).strftime("%H:%M")
            lines.append(f"\n🌅 Sonnenaufgang {sr} · 🌇 Sonnenuntergang {ss}")
        except (ValueError, IndexError):
            pass

    return SkillResult(
        inhalt="\n".join(lines),
        typ="daten",
        metadaten={"stadt": display, "lat": lat, "lon": lon, "quelle": "Open-Meteo"},
    )
