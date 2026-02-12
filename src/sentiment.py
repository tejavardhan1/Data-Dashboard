"""Sentiment analysis for news and text using TextBlob or fallback."""

import re

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

MISINFO_KEYWORDS = ["deepfake", "fake news", "misinformation", "disinformation", "AI-generated", "AI-generated content", "hoax", "false claim", "manipulated", "synthetic media", "phishing"]
TRUSTED_SOURCES = ["reuters", "ap", "bbc", "nytimes", "wsj", "bloomberg", "npr", "associated press"]
SUSPICIOUS_PATTERNS = ["urgent", "click here", "act now", "verified by ai", "100% real"]


def authenticity_score(text: str, source: str = "") -> float:
    """Return 0-100 authenticity score. Higher = more likely authentic."""
    if not text:
        return 50.0
    t, s = str(text).lower(), str(source).lower()
    score = 70.0
    if any(ts in s for ts in TRUSTED_SOURCES):
        score += 15
    if any(kw in t for kw in MISINFO_KEYWORDS):
        score -= 30
    if any(p in t for p in SUSPICIOUS_PATTERNS):
        score -= 15
    return max(0, min(100, round(score, 0)))


def deepfake_risk_score(text: str) -> float:
    """Return 0-100 deepfake/misinfo risk. Higher = higher risk."""
    if not text:
        return 0.0
    t = str(text).lower()
    hits = sum(1 for kw in MISINFO_KEYWORDS if kw in t)
    susp = sum(1 for p in SUSPICIOUS_PATTERNS if p in t)
    return min(100, round(hits * 15 + susp * 10, 0))


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
