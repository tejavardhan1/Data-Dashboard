import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from api_fetcher import fetch_all_news, fetch_all_weather, fetch_crypto_yahoo, fetch_stock_history, fetch_crypto_history, fetch_yahoo_finance
from data_processing import aggregate_forecast_by_day, clean_crypto_data, clean_stock_data, compute_trend_alerts
from ml_model import get_predictions_summary

st.set_page_config(page_title="Real-Time Multi-API Dashboard", page_icon="📊", layout="wide")


def bar_chart(df, x, y, colors):
    fig = px.bar(df, x=x, y=y, color=y, color_continuous_scale=colors, color_continuous_midpoint=0)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig


st.markdown("""
<style>
.alert-high { background: #3d1f1f; border-left: 4px solid #f85149; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0; }
.alert-medium { background: #2d2a1f; border-left: 4px solid #d29922; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


def main():
    st.sidebar.header("⚙️ Controls")
    city = st.sidebar.text_input("City", "London")
    country = st.sidebar.text_input("Country", "UK")
    news_query = st.sidebar.text_input("News search", "technology")
    symbols_str = st.sidebar.text_input("Stocks (comma)", "AAPL,GOOGL,MSFT,AMZN,META")
    symbols = [s.strip() for s in symbols_str.split(",") if s.strip()] or ["AAPL", "GOOGL", "MSFT"]
    refresh_interval = st.sidebar.slider("Auto-refresh (seconds)", 30, 300, 60, 30)

    st_autorefresh(interval=refresh_interval * 1000, key="autorefresh")

    location_key = f"{city}|{country}|{news_query}"
    last_ts = st.session_state.get("last_refresh_ts", 0)
    elapsed = (datetime.now().timestamp() - last_ts) if last_ts else refresh_interval + 1
    needs_refresh = (
        "finance_data" not in st.session_state
        or st.session_state.get("cache_key") != location_key
        or elapsed >= refresh_interval
        or st.sidebar.button("🔄 Refresh Now")
    )
    if needs_refresh:
        with st.spinner("Fetching..."):
            st.session_state.finance_data = {"stocks": fetch_yahoo_finance(symbols), "crypto": fetch_crypto_yahoo()}
            st.session_state.weather_data = fetch_all_weather(city, country)
            st.session_state.cache_key = location_key
            st.session_state.news_data = fetch_all_news([news_query] if news_query else None)
            st.session_state.last_refresh = datetime.now().strftime("%H:%M:%S")
            st.session_state.last_refresh_ts = datetime.now().timestamp()

    st.title("📊 Real-Time Multi-API Dashboard")
    if "last_refresh" in st.session_state:
        st.caption(f"Last refresh: {st.session_state.last_refresh}")

    finance = st.session_state.get("finance_data", {})
    stocks = clean_stock_data(finance.get("stocks", pd.DataFrame()))
    crypto = clean_crypto_data(finance.get("crypto", pd.DataFrame()))

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Finance", "🌤️ Weather", "📰 News", "🔮 Predictions"])

    with tab1:
        if stocks.empty and crypto.empty:
            st.info("No finance data. Try again in a moment.")
        else:
            alerts = compute_trend_alerts(stocks, crypto)
            for a in alerts[:5]:
                cls = "alert-high" if a["severity"] == "high" else "alert-medium"
                st.markdown(f'<div class="{cls}">{a["message"]}</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📈 Stocks")
                if not stocks.empty:
                    st.plotly_chart(bar_chart(stocks, "symbol", "change", ["#f85149", "#8b949e", "#3fb950"]), use_container_width=True)
                    df = stocks[["symbol", "price", "change", "volume"]].copy()
                    df["price"] = df["price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "")
                    df["change"] = df["change"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
                    df["volume"] = df["volume"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
                    st.dataframe(df, use_container_width=True, hide_index=True)
            with c2:
                st.subheader("₿ Crypto")
                if not crypto.empty:
                    st.plotly_chart(bar_chart(crypto, "symbol", "change_24h", ["#f85149", "#8b949e", "#3fb950"]), use_container_width=True)
                    df = crypto[["symbol", "price", "change_24h"]].copy()
                    df["price"] = df["price"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
                    df["change_24h"] = df["change_24h"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
                    st.dataframe(df, use_container_width=True, hide_index=True)

    with tab2:
        w = st.session_state.get("weather_data", {})
        current = w.get("current")
        forecast = w.get("forecast", pd.DataFrame())
        if not current:
            st.warning("Could not load weather for this location")
        else:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader(f"🌤️ {current.get('city')}, {current.get('country')}")
                st.metric("Temp", f"{current.get('temp', 0):.1f}°C")
                st.metric("Feels like", f"{current.get('feels_like', 0):.1f}°C")
                st.metric("Humidity", f"{current.get('humidity', 0)}%")
            with c2:
                if not forecast.empty:
                    agg = aggregate_forecast_by_day(forecast)
                    if not agg.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=agg["date"].astype(str), y=agg["temp_max"], name="Max", marker_color="#58a6ff"))
                        fig.add_trace(go.Bar(x=agg["date"].astype(str), y=agg["temp_mean"], name="Mean", marker_color="#8b949e"))
                        fig.add_trace(go.Bar(x=agg["date"].astype(str), y=agg["temp_min"], name="Min", marker_color="#388bfd"))
                        fig.update_layout(barmode="group", template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        news = st.session_state.get("news_data", pd.DataFrame())
        if news.empty:
            st.warning("Add NEWS_API_KEY to .env")
        else:
            for _, row in news.head(15).iterrows():
                with st.expander(f"**{row.get('title', '')}** ({row.get('source', '')})"):
                    st.caption(row.get("published_at"))
                    st.write(row.get("description"))
                    if row.get("url"):
                        st.link_button("Read", row["url"])

    with tab4:
        stock_hist = {s: fetch_stock_history(s) for s in stocks["symbol"].tolist()} if not stocks.empty else {}
        crypto_hist = {s: fetch_crypto_history(s) for s in crypto["symbol"].tolist()} if not crypto.empty else {}
        summary = get_predictions_summary(
            stocks, crypto,
            st.session_state.get("weather_data", {}).get("forecast", pd.DataFrame()),
            stock_hist, crypto_hist,
        )
        st.caption("Linear regression on 30-day history • Predicted next-day price")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("📈 Stocks")
            for s in summary.get("stocks", []):
                with st.container():
                    ch = s.get("change_pct", 0)
                    st.metric(s["symbol"], f"${s['current']:.2f}", f"{ch:+.2f}% → ${s.get('predicted', 0):.2f}")
                    with st.expander("Details"):
                        st.write(f"**Trend:** {s.get('trend', 'flat')} • **Volatility:** {s.get('volatility', 0):.1f}%")
                        st.write(f"**5d MA:** ${s.get('ma5', 0):.2f} • **7d range:** ${s.get('range_7d', (0,0))[0]:.2f} - ${s.get('range_7d', (0,0))[1]:.2f}")
                        if s.get("pred_3d"):
                            st.write(f"**Pred 3d:** ${s['pred_3d'][-1]:.2f}")
                        if s.get("pred_5d"):
                            st.write(f"**Pred 5d:** ${s['pred_5d'][-1]:.2f}")

        with c2:
            st.subheader("₿ Crypto")
            for c in summary.get("crypto", []):
                ch = c.get("change_pct", 0)
                st.metric(c["symbol"], f"${c['current']:,.2f}", f"{ch:+.2f}% → ${c.get('predicted', 0):,.2f}")
                with st.expander(f"{c['symbol']} details"):
                    st.write(f"**Trend:** {c.get('trend', 'flat')} • **Volatility:** {c.get('volatility', 0):.1f}%")
                    st.write(f"**5d MA:** ${c.get('ma5', 0):,.2f} • **7d range:** ${c.get('range_7d', (0,0))[0]:,.0f} - ${c.get('range_7d', (0,0))[1]:,.0f}")
                    if c.get("pred_3d"):
                        st.write(f"**Pred 3d:** ${c['pred_3d'][-1]:,.2f}")
                    if c.get("pred_5d"):
                        st.write(f"**Pred 5d:** ${c['pred_5d'][-1]:,.2f}")

        with c3:
            st.subheader("🌤️ Weather")
            if summary.get("weather"):
                w = summary["weather"]
                preds = w.get("predicted_temps", [])
                st.write(f"**Trend:** {w.get('trend', 'flat')}")
                if preds:
                    st.write("**Predicted temps (next 5 periods):**")
                    for i, t in enumerate(preds, 1):
                        st.write(f"  Day {i}: {t:.1f}°C" if isinstance(t, (int, float)) else f"  Day {i}: {t}")


if __name__ == "__main__":
    main()
