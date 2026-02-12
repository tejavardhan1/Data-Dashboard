from typing import Optional

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def predict_trend(values: list[float], horizon: int = 1) -> Optional[list[float]]:
    if not SKLEARN_AVAILABLE or len(values) < 5:
        return None
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 5:
        return None
    X = np.arange(len(vals)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, vals)
    next_X = np.arange(len(vals), len(vals) + horizon).reshape(-1, 1)
    return model.predict(next_X).tolist()


def moving_avg(values: list[float], window: int = 5, horizon: int = 1) -> list[float]:
    if not values or len(values) < window:
        return []
    avg = sum(values[-window:]) / window
    return [avg] * horizon


def trend_direction(values: list[float]) -> str:
    if not values or len(values) < 2:
        return "flat"
    m1 = sum(values[: len(values) // 2]) / (len(values) // 2)
    m2 = sum(values[len(values) // 2 :]) / (len(values) - len(values) // 2)
    diff = (m2 - m1) / m1 * 100 if m1 else 0
    return "up" if diff > 1 else "down" if diff < -1 else "flat"


def _volatility(values: list[float]) -> float:
    if not values or len(values) < 2:
        return 0.0
    return float(np.std(values) / np.mean(values) * 100) if np.mean(values) else 0.0


def _recent_range(values: list[float], days: int = 7) -> tuple[float, float]:
    if not values or len(values) < days:
        return (min(values), max(values)) if values else (0.0, 0.0)
    v = values[-days:]
    return float(min(v)), float(max(v))


def get_predictions_summary(stocks_df: pd.DataFrame, crypto_df: pd.DataFrame, forecast_df: pd.DataFrame, stock_history: dict = None, crypto_history: dict = None) -> dict:
    summary = {"stocks": [], "crypto": [], "weather": []}
    stock_history = stock_history or {}
    crypto_history = crypto_history or {}

    def _add_pred(sym: str, current: float, prices: list, fallback: list) -> dict:
        pred_1d = predict_trend(prices, 1) if len(prices) >= 5 else predict_trend(fallback, 1)
        pred_3d = predict_trend(prices, 3) if len(prices) >= 5 else None
        pred_5d = predict_trend(prices, 5) if len(prices) >= 5 else None
        pred_val = pred_1d[0] if pred_1d else current
        change_pct = ((pred_val - current) / current * 100) if current else 0
        ma5 = sum(prices[-5:]) / min(5, len(prices)) if prices else current
        low, high = _recent_range(prices, 7) if len(prices) >= 7 else (current, current)
        return {
            "symbol": sym, "current": current, "predicted": pred_val,
            "pred_3d": pred_3d, "pred_5d": pred_5d,
            "change_pct": change_pct, "trend": trend_direction(prices if len(prices) >= 2 else [current, current]),
            "volatility": _volatility(prices), "ma5": ma5,
            "range_7d": (low, high),
        }

    if SKLEARN_AVAILABLE and stocks_df is not None and not stocks_df.empty:
        for _, row in stocks_df.iterrows():
            p, sym = row.get("price"), row.get("symbol")
            if p and pd.notna(p):
                prices = stock_history.get(sym, [])
                item = _add_pred(sym, float(p), prices, [float(p)] * 5 + [float(p) * 1.01])
                summary["stocks"].append(item)
    if SKLEARN_AVAILABLE and crypto_df is not None and not crypto_df.empty:
        for _, row in crypto_df.iterrows():
            p, sym = row.get("price"), row.get("symbol")
            if p and pd.notna(p):
                prices = crypto_history.get(sym, [])
                item = _add_pred(sym, float(p), prices, [float(p) * 0.98, float(p) * 0.99, float(p)] * 2)
                summary["crypto"].append(item)
    if forecast_df is not None and not forecast_df.empty and "temp" in forecast_df.columns:
        temps = forecast_df["temp"].dropna().tolist()
        if temps:
            pred = predict_trend(temps[-24:], 5) if SKLEARN_AVAILABLE else moving_avg(temps, 6, 5)
            summary["weather"] = {"predicted_temps": pred or moving_avg(temps, 6, 5), "trend": trend_direction(temps)}
    return summary
