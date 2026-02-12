"""Sentiment analysis for news and text using TextBlob or fallback."""

import re

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

MISINFO_KEYWORDS = ["deepfake", "fake news", "misinformation", "disinformation", "AI-generated", "hoax", "false claim"]


def score_sentiment(text: str) -> float:
    """Return sentiment score -1 (negative) to 1 (positive)."""
    if not text or not str(text).strip():
        return 0.0
    text = str(text)[:500]
    if TEXTBLOB_AVAILABLE:
        try:
            return float(TextBlob(text).sentiment.polarity)
        except Exception:
            pass
    neg = len(re.findall(r"\b(?:crash|drop|fall|loss|risk|fear|warn|fail|decline|recession)\b", text, re.I))
    pos = len(re.findall(r"\b(?:surge|rise|gain|growth|rally|recovery|breakthrough|optimism)\b", text, re.I))
    total = neg + pos or 1
    return round((pos - neg) / total, 2)


def flag_misinfo_risk(text: str) -> bool:
    """Flag text that may relate to AI misinformation or deepfakes."""
    if not text:
        return False
    t = str(text).lower()
    return any(kw in t for kw in MISINFO_KEYWORDS)


def aggregate_sentiment(scores: list[float]) -> dict:
    """Return mean sentiment and stress level (0-100)."""
    if not scores:
        return {"mean": 0.0, "stress": 50}
    mean = sum(scores) / len(scores)
    stress = max(0, min(100, 50 - mean * 50))
    return {"mean": round(mean, 2), "stress": round(stress, 0)}
