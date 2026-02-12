from typing import Optional

import pandas as pd


def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out = out.dropna(subset=["price"], how="all")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["change"] = pd.to_numeric(out["change"], errors="coerce").fillna(0)
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0)
    return out


def clean_crypto_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out = out.dropna(subset=["price"], how="all")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["change_24h"] = pd.to_numeric(out.get("change_24h", 0), errors="coerce").fillna(0)
    return out


def compute_trend_alerts(stocks_df: pd.DataFrame, crypto_df: pd.DataFrame) -> list[dict]:
    alerts = []
    for _, row in stocks_df.iterrows():
        ch = row.get("change", 0) or 0
        if abs(ch) > 5:
            alerts.append({"message": f"{row.get('symbol')} moved {ch:.2f}%", "severity": "high" if abs(ch) > 10 else "medium"})
    for _, row in crypto_df.iterrows():
        ch = row.get("change_24h", 0) or 0
        if abs(ch) > 8:
            alerts.append({"message": f"{row.get('symbol')} moved {ch:.2f}% (24h)", "severity": "high" if abs(ch) > 15 else "medium"})
    return alerts


def aggregate_forecast_by_day(forecast_df: pd.DataFrame) -> pd.DataFrame:
    if forecast_df.empty or "dt" not in forecast_df.columns:
        return pd.DataFrame()
    df = forecast_df.copy()
    df["date"] = pd.to_datetime(df["dt"]).dt.date
    if "temp_min" in df.columns and "temp_max" in df.columns:
        return df.groupby("date").agg(temp_min=("temp_min", "min"), temp_max=("temp_max", "max"), temp_mean=("temp", "mean")).reset_index()
    return df.groupby("date").agg(temp_min=("temp", "min"), temp_max=("temp", "max"), temp_mean=("temp", "mean")).reset_index()
