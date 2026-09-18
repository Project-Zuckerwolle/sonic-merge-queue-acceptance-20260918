"""Nova Predator v1 Skill — Finanzen.

Kurse via:
  - Yahoo Finance JSON API (Aktien, Indizes, Krypto, FX) — kein Key nötig
  - CoinGecko API (Krypto) — kein Key nötig
"""
from __future__ import annotations

import json
import re
import urllib.request
import urllib.parse
from dataclasses import dataclass

from core.logger import get
from core.skill_registry import SkillContext, SkillResult

log = get("skill.finanzen")


@dataclass
class Kurs:
    symbol: str
    name: str
    preis: float
    waehrung: str
    aenderung_pct: float | None
    aenderung_abs: float | None
    marktkap: str = ""


# Bekannte Ticker-Erkennungen aus User-Input
BEKANNTE_TICKER: dict[str, str] = {
    "dax": "^GDAXI", "dow": "^DJI", "dow jones": "^DJI",
    "s&p": "^GSPC", "sp500": "^GSPC", "nasdaq": "^IXIC",
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
    "amazon": "AMZN", "tesla": "TSLA", "nvidia": "NVDA",
    "meta": "META", "netflix": "NFLX", "adobe": "ADBE",
    "eur usd": "EURUSD=X", "usd eur": "EURUSD=X",
    "euro dollar": "EURUSD=X", "dollar euro": "EURUSD=X",
    "eur gbp": "EURGBP=X", "eur chf": "EURCHF=X",
    "gold": "GC=F", "silber": "SI=F", "öl": "CL=F", "oil": "CL=F",
    "bitcoin": "BTC-EUR", "btc": "BTC-EUR",
    "ethereum": "ETH-EUR", "eth": "ETH-EUR",
    "solana": "SOL-EUR", "sol": "SOL-EUR",
}

KRYPTO_NAMEN: dict[str, str] = {
    "bitcoin": "bitcoin", "btc": "bitcoin",
    "ethereum": "ethereum", "eth": "ethereum",
    "solana": "solana", "sol": "solana",
    "cardano": "cardano", "ada": "cardano",
    "dogecoin": "dogecoin", "doge": "dogecoin",
    "xrp": "ripple", "ripple": "ripple",
}


def _yahoo_kurs(symbol: str) -> Kurs | None:
    """Ruft Kurs von Yahoo Finance ab."""
    sym_enc = urllib.parse.quote(symbol)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{sym_enc}"
        f"?interval=1d&range=1d"
    )
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 Nova-v10/1.0",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())

        result = data["chart"]["result"][0]
        meta   = result["meta"]
        preis  = meta.get("regularMarketPrice") or meta.get("previousClose", 0)
        prev   = meta.get("chartPreviousClose") or meta.get("previousClose", preis)
        waehr  = meta.get("currency", "USD")
        name   = meta.get("shortName") or meta.get("longName") or symbol

        aend_abs = preis - prev if prev else None
        aend_pct = (aend_abs / prev * 100) if prev and aend_abs is not None else None

        return Kurs(
            symbol=symbol,
            name=name,
            preis=preis,
            waehrung=waehr,
            aenderung_pct=aend_pct,
            aenderung_abs=aend_abs,
        )
    except Exception:
        return None


def _coingecko_kurs(coin_id: str, vs_currency: str = "eur") -> Kurs | None:
    """Ruft Kurs von CoinGecko ab."""
    url = (
        f"https://api.coingecko.com/api/v3/simple/price"
        f"?ids={coin_id}&vs_currencies={vs_currency}"
        f"&include_24hr_change=true&include_market_cap=true"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Nova-v10/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())

        if coin_id not in data:
            return None

        d       = data[coin_id]
        preis   = d.get(vs_currency, 0)
        aend_pct = d.get(f"{vs_currency}_24h_change")
        mkcap   = d.get(f"{vs_currency}_market_cap", 0)

        def _mrd(v: float) -> str:
            if v >= 1e12: return f"{v/1e12:.1f}B €"
            if v >= 1e9:  return f"{v/1e9:.1f}Mrd €"
            if v >= 1e6:  return f"{v/1e6:.0f}Mio €"
            return f"{v:.0f} €"

        return Kurs(
            symbol=coin_id.upper(),
            name=coin_id.capitalize(),
            preis=preis,
            waehrung=vs_currency.upper(),
            aenderung_pct=aend_pct,
            aenderung_abs=None,
            marktkap=_mrd(mkcap) if mkcap else "",
        )
    except Exception:
        return None


def _symbol_aus_input(user_input: str) -> list[str]:
    """Extrahiert Ticker-Symbole oder erkannte Begriffe aus Input."""
    text = user_input.lower()
    gefunden: list[str] = []

    # Bekannte Begriffe prüfen
    for schluessel, symbol in BEKANNTE_TICKER.items():
        if schluessel in text:
            gefunden.append(symbol)

    # Explizite Ticker (Großbuchstaben-Patterns) z.B. "AAPL", "ETH-EUR"
    # Wörter ausschließen die kein Ticker sind
    kein_ticker = {"EUR", "USD", "GBP", "CHF", "JPY", "CAD", "AUD",
                   "DAX", "DOW", "BTC", "ETH", "SOL", "KI", "AI",
                   "RSS", "API", "URL", "HTML", "CSS", "SQL"}
    ticker_pattern = re.findall(r'\b([A-Z]{2,6}(?:-[A-Z]{2,4})?)\b', user_input)
    for t in ticker_pattern:
        if t not in kein_ticker and t not in gefunden:
            gefunden.append(t)

    return list(dict.fromkeys(gefunden))[:4]  # Deduplizieren, max 4


def _ist_krypto(symbol: str) -> str | None:
    """Gibt CoinGecko-ID zurück wenn es eine Krypto ist."""
    sym_lower = symbol.lower().replace("-eur", "").replace("-usd", "")
    return KRYPTO_NAMEN.get(sym_lower)


def _format_kurs(k: Kurs) -> str:
    """Formatiert einen Kurs als lesbare Zeile."""
    # Preis formatieren
    if k.preis >= 1000:
        preis_str = f"{k.preis:,.2f}"
    elif k.preis >= 1:
        preis_str = f"{k.preis:.4f}"
    else:
        preis_str = f"{k.preis:.6f}"

    # Änderung
    if k.aenderung_pct is not None:
        pfeil  = "▲" if k.aenderung_pct >= 0 else "▼"
        farbe  = "+" if k.aenderung_pct >= 0 else ""
        aend   = f" {pfeil} {farbe}{k.aenderung_pct:.2f}%"
    else:
        aend = ""

    # Marktkapitalisierung
    mkap = f" · Mktcap {k.marktkap}" if k.marktkap else ""

    return f"**{k.name}** ({k.symbol}): {preis_str} {k.waehrung}{aend}{mkap}"


def tool_definition() -> dict:
    """Tool-Definition für Gemma Tool-Calling (v3)."""
    return {
        "type": "function",
        "function": {
            "name": "finanzen",
            "description": "Aktuelle Kurse für Aktien, Krypto, Indizes und Wechselkurse. Nutze dieses Tool bei Fragen nach Börsenkursen, Bitcoin, DAX, Aktienpreisen oder Währungskursen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbole": {
                        "type": "string",
                        "description": "Komma-getrennte Symbole oder Namen, z.B. 'Bitcoin, NVIDIA, DAX' oder 'EUR USD Kurs'",
                    },
                },
                "required": ["symbole"],
            },
        },
    }


def on_message(ctx: SkillContext) -> SkillResult | None:
    cfg = ctx.skill_config

    symbole = _symbol_aus_input(ctx.user_input)

    # Wenn keine spezifischen Symbole erkannt → Standard-Symbole aus Config
    if not symbole:
        symbole = cfg.get("standard_symbole", ["^GDAXI", "BTC-EUR"])

    log.info("Skill 'finanzen' aufgerufen: symbole=%s", symbole)

    lines: list[str] = ["**Finanzdaten**\n"]
    gefunden = 0

    for sym in symbole:
        # Krypto via CoinGecko versuchen
        coin_id = _ist_krypto(sym)
        if coin_id:
            k = _coingecko_kurs(coin_id)
            if k:
                lines.append(_format_kurs(k))
                gefunden += 1
                continue

        # Alles andere via Yahoo Finance
        k = _yahoo_kurs(sym)
        if k:
            lines.append(_format_kurs(k))
            gefunden += 1
        else:
            lines.append(f"⚠️ {sym}: Kurs nicht verfügbar")

    if gefunden == 0:
        return SkillResult(
            inhalt="❌ Keine Finanzdaten verfügbar (Netzwerk prüfen?).",
            typ="fehler",
        )

    lines.append("\n*Kurse können verzögert sein (15 min).*")

    return SkillResult(
        inhalt="\n".join(lines),
        typ="daten",
        metadaten={"symbole": symbole, "quelle": "Yahoo Finance / CoinGecko"},
    )
