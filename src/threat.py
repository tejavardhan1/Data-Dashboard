"""Cybersecurity threat analysis and threat feed integration."""

import os
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

THREAT_KEYWORDS = {
    "phishing": ["phish", "credential", "password steal", "login fake", "suspicious link"],
    "malware": ["malware", "ransomware", "trojan", "virus", "exploit", "payload"],
    "breach": ["data breach", "leak", "compromised", "hacked", "exfiltrat"],
    "ddos": ["ddos", "denial of service", "botnet", "flood attack"],
    "vulnerability": ["cve-", "zero-day", "vulnerability", "patch"],
}


def classify_threat(text: str) -> list[str]:
    """Classify text into threat categories using keyword matching."""
    if not text:
        return []
    t = str(text).lower()
    cats = []
    for cat, kws in THREAT_KEYWORDS.items():
        if any(kw in t for kw in kws):
            cats.append(cat)
    return cats


def fetch_abuseipdb_reports(ip: str = None, limit: int = 10) -> pd.DataFrame:
    """Fetch recent reports from AbuseIPDB (requires API key)."""
    key = os.getenv("ABUSEIPDB_API_KEY")
    if not key or not ip:
        return pd.DataFrame()
    try:
        r = requests.get(
            "https://api.abuseipdb.com/api/v2/check",
            params={"ipAddress": ip},
            headers={"Key": key.strip(), "Accept": "application/json"},
            timeout=10,
        )
        if r.status_code == 200:
            d = r.json().get("data", {})
            return pd.DataFrame([{
                "ip": d.get("ipAddress"),
                "confidence": d.get("abuseConfidenceScore", 0),
                "country": d.get("countryCode"),
                "domain": d.get("domain"),
                "reports": d.get("totalReports", 0),
                "categories": ", ".join(str(c) for c in d.get("reports", [{}])[:3]),
            }])
    except Exception:
        pass
    return pd.DataFrame()


def fetch_threat_news() -> pd.DataFrame:
    """Fetch cybersecurity news via News API."""
    key = os.getenv("NEWS_API_KEY")
    if not key:
        return _demo_threats()
    try:
        from datetime import datetime, timedelta
        from api_fetcher import fetch_news_api
        df = fetch_news_api("cybersecurity data breach ransomware phishing", page_size=10)
        if df.empty:
            return pd.DataFrame()
        df["threat_cats"] = (df.get("title", "") + " " + df.get("description", "")).fillna("").apply(classify_threat)
        df["threat_level"] = df["threat_cats"].apply(lambda x: "High" if len(x) >= 2 else "Medium" if x else "Low")
        return df if not df.empty else _demo_threats()
    except Exception:
        return _demo_threats()


def _demo_threats() -> pd.DataFrame:
    """Demo threat alerts when no API."""
    return pd.DataFrame([
        {"title": "Ransomware campaign targets healthcare sector", "source": "Threat Intel", "threat_cats": ["ransomware", "breach"], "threat_level": "High"},
        {"title": "New phishing kit impersonates cloud providers", "source": "Threat Intel", "threat_cats": ["phishing"], "threat_level": "Medium"},
        {"title": "CVE-2024-XXXX critical vulnerability disclosed", "source": "Threat Intel", "threat_cats": ["vulnerability"], "threat_level": "Medium"},
    ])
