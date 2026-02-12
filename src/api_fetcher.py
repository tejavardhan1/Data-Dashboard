import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def fetch_yahoo_finance(symbols: list[str] = None) -> pd.DataFrame:
    symbols = symbols or ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
    rows = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d", interval="1h")
            if hist.empty:
                info = ticker.info
                rows.append({
                    "symbol": symbol,
                    "price": info.get("regularMarketPrice") or info.get("previousClose"),
                    "change": info.get("regularMarketChangePercent", 0),
                    "volume": info.get("regularMarketVolume", 0),
                })
            else:
                latest = hist.iloc[-1]
                prev = hist.iloc[-2] if len(hist) > 1 else latest
                ch = ((latest["Close"] - prev["Close"]) / prev["Close"] * 100) if prev["Close"] else 0
                rows.append({"symbol": symbol, "price": float(latest["Close"]), "change": ch, "volume": int(latest["Volume"])})
        except Exception:
            rows.append({"symbol": symbol, "price": None, "change": 0, "volume": 0})
    return pd.DataFrame(rows)


def fetch_crypto_yahoo(symbols: list[str] = None) -> pd.DataFrame:
    symbols = symbols or ["BTC-USD", "ETH-USD", "SOL-USD"]
    rows = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d", interval="1h")
            info = ticker.info
            if hist.empty:
                rows.append({
                    "symbol": symbol.replace("-USD", ""),
                    "price": info.get("regularMarketPrice") or info.get("previousClose"),
                    "change_24h": info.get("regularMarketChangePercent", 0),
                })
            else:
                latest = hist.iloc[-1]
                prev = hist.iloc[-24] if len(hist) >= 24 else hist.iloc[0]
                ch = ((latest["Close"] - prev["Close"]) / prev["Close"] * 100) if prev["Close"] else 0
                rows.append({"symbol": symbol.replace("-USD", ""), "price": float(latest["Close"]), "change_24h": ch})
        except Exception:
            rows.append({"symbol": symbol.replace("-USD", ""), "price": None, "change_24h": 0})
    return pd.DataFrame(rows)


WMO_CODES = {0: "Clear", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast", 45: "Fog", 48: "Rime Fog",
    51: "Light Drizzle", 53: "Drizzle", 55: "Dense Drizzle", 61: "Slight Rain", 63: "Rain", 65: "Heavy Rain",
    71: "Slight Snow", 73: "Snow", 75: "Heavy Snow", 80: "Slight Showers", 81: "Showers", 82: "Heavy Showers", 95: "Thunderstorm"}


COUNTRY_MAP = {"UK": "United Kingdom", "US": "United States", "USA": "United States", "UAE": "United Arab Emirates"}


def _geocode(city: str, country: str) -> tuple[float, float, str, str] | None:
    def _try(q: str):
        try:
            r = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": q, "count": 5}, timeout=8)
            if r.status_code == 200:
                data = r.json()
                results = data.get("results") or []
                if results and country:
                    country_full = COUNTRY_MAP.get(country.upper(), country)
                    for res in results:
                        if res.get("country", "").lower() == country_full.lower() or res.get("country_code", "").upper() == country.upper()[:2]:
                            return res["latitude"], res["longitude"], res.get("name", city), res.get("country", country)
                if results:
                    res = results[0]
                    return res["latitude"], res["longitude"], res.get("name", city), res.get("country", country)
        except Exception:
            pass
        return None
    if country:
        q = f"{city},{COUNTRY_MAP.get(country.upper(), country)}"
        result = _try(q)
        if result:
            return result
    return _try(city)


def fetch_weather(city: str = "London", country_code: str = "UK", lat: float = None, lon: float = None):
    try:
        disp_city, disp_country = (f"{lat:.2f}, {lon:.2f}", "") if (lat is not None and lon is not None) else (city, country_code)
        if lat is None or lon is None:
            geo = _geocode(city, country_code)
            if not geo:
                return {"city": city, "country": country_code, "temp": 0, "feels_like": 0, "humidity": 0, "wind_speed": 0, "description": "Unknown location"}
            lat, lon, disp_city, disp_country = geo
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=10,
        )
        if r.status_code == 200:
            d = r.json()
            c = d.get("current", {})
            code = int(c.get("weather_code", 0))
            return {
                "city": disp_city,
                "country": disp_country or "",
                "temp": c.get("temperature_2m"),
                "feels_like": c.get("apparent_temperature"),
                "humidity": c.get("relative_humidity_2m"),
                "wind_speed": c.get("wind_speed_10m"),
                "description": WMO_CODES.get(code, "Clear"),
            }
    except Exception:
        pass
    return {"city": city, "country": country_code, "temp": 0, "feels_like": 0, "humidity": 0, "wind_speed": 0, "description": "Error"}


def fetch_weather_forecast(city: str = "London", country_code: str = "UK", lat: float = None, lon: float = None) -> pd.DataFrame:
    try:
        if lat is None or lon is None:
            geo = _geocode(city, country_code)
            if not geo:
                return pd.DataFrame()
            lat, lon, _, _ = geo
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,weather_code",
                "timezone": "auto",
                "forecast_days": 7,
            },
            timeout=10,
        )
        if r.status_code == 200:
            d = r.json()
            daily = d.get("daily", {})
            times = daily.get("time", [])
            tmax = daily.get("temperature_2m_max", [])
            tmin = daily.get("temperature_2m_min", [])
            tmean = daily.get("temperature_2m_mean", [])
            codes = daily.get("weather_code", [])
            rows = []
            for i in range(min(len(times), 7)):
                tm = tmean[i] if i < len(tmean) else ((tmax[i] + tmin[i]) / 2 if i < len(tmax) and i < len(tmin) else 0)
                rows.append({
                    "dt": times[i] + " 12:00",
                    "temp": tm,
                    "temp_min": tmin[i] if i < len(tmin) else tm,
                    "temp_max": tmax[i] if i < len(tmax) else tm,
                    "feels_like": tm,
                    "humidity": 0,
                    "description": WMO_CODES.get(int(codes[i]) if i < len(codes) else 0, "Clear"),
                    "wind_speed": 0,
                })
            return pd.DataFrame(rows)
    except Exception:
        pass
    return pd.DataFrame()


def _demo_news() -> pd.DataFrame:
    return pd.DataFrame([
        {"title": "Tech stocks rally on AI optimism (demo)", "source": "Demo", "description": "Add NEWS_API_KEY to .env for live headlines.", "url": "", "published_at": ""},
        {"title": "Crypto markets show volatility (demo)", "source": "Demo", "description": "Get a free key at newsapi.org", "url": "", "published_at": ""},
        {"title": "Weather patterns shift across regions (demo)", "source": "Demo", "description": "Placeholder - configure API keys for real data.", "url": "", "published_at": ""},
    ])


def fetch_news_api(query: str, page_size: int = 10) -> pd.DataFrame:
    key = os.getenv("NEWS_API_KEY") or os.getenv("news_api_key")
    if not key or key == "your_news_api_key":
        return pd.DataFrame()
    try:
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": query, "from": from_date, "sortBy": "publishedAt", "pageSize": page_size, "apiKey": key.strip()},
            timeout=15,
        )
        data = r.json()
        if data.get("status") == "error":
            return pd.DataFrame()
        articles = data.get("articles", [])
        if not articles:
            return pd.DataFrame()
        rows = []
        for a in articles:
            rows.append({
                "title": a.get("title", ""),
                "source": a.get("source", {}).get("name", ""),
                "description": (a.get("description") or "")[:200],
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
            })
        return pd.DataFrame(rows)
    except Exception:
        pass
    return pd.DataFrame()


def fetch_stock_history(symbol: str, days: int = 30) -> list[float]:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{days}d", interval="1d")
        if hist.empty or len(hist) < 5:
            return []
        return hist["Close"].dropna().tolist()
    except Exception:
        return []


def fetch_crypto_history(symbol: str, days: int = 30) -> list[float]:
    sym = symbol if "-USD" in symbol else f"{symbol}-USD"
    try:
        ticker = yf.Ticker(sym)
        hist = ticker.history(period=f"{days}d", interval="1d")
        if hist.empty or len(hist) < 5:
            return []
        return hist["Close"].dropna().tolist()
    except Exception:
        return []


def fetch_all_finance() -> dict[str, pd.DataFrame]:
    return {"stocks": fetch_yahoo_finance(), "crypto": fetch_crypto_yahoo()}


def fetch_all_weather(city: str = "London", country: str = "UK", lat: float = None, lon: float = None) -> dict[str, Any]:
    return {"current": fetch_weather(city, country, lat, lon), "forecast": fetch_weather_forecast(city, country, lat, lon)}


def fetch_all_news(queries: list[str] = None) -> pd.DataFrame:
    queries = queries or ["technology"]
    dfs = [fetch_news_api(q) for q in queries]
    result = pd.concat([d for d in dfs if not d.empty], ignore_index=True) if any(not d.empty for d in dfs) else pd.DataFrame()
    return result if not result.empty else _demo_news()
