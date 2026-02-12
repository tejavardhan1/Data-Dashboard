import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from api_fetcher import fetch_all_news, fetch_all_weather, fetch_crypto_yahoo, fetch_crypto_history, fetch_reddit_sentiment, fetch_stock_history, fetch_yahoo_finance
from data_processing import add_sentiment_to_news, aggregate_forecast_by_day, clean_crypto_data, clean_stock_data, compute_trend_alerts, market_stress_score
from ml_model import get_predictions_summary
from sentiment import aggregate_sentiment
from anomaly import zscore_anomaly
from threat import fetch_threat_news, classify_threat, fetch_abuseipdb_reports

st.set_page_config(page_title="Insight Dashboard", page_icon="◈", layout="wide", initial_sidebar_state="expanded")


def bar_chart(df, x, y, colors):
    fig = px.bar(df, x=x, y=y, color=y, color_continuous_scale=colors, color_continuous_midpoint=0)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig


def line_chart(df, x, y, title=""):
    fig = px.line(df, x=x, y=y, markers=True, title=title)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
    fig.update_traces(line=dict(width=2))
    return fig


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

.stApp { background: linear-gradient(180deg, #0a0e14 0%, #0d1117 30%, #161b22 100%); }
.main .block-container { padding: 2rem 3rem; max-width: 100%; }
h1, h2, h3 { font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; letter-spacing: -0.02em; }
h1 { font-size: 1.9rem !important; background: linear-gradient(135deg, #e6edf3 0%, #8b949e 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
h2 { font-size: 1.2rem !important; color: #c9d1d9 !important; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem; margin-top: 1.5rem; }

.card { background: linear-gradient(145deg, #161b22 0%, #21262d 100%); border: 1px solid #30363d; border-radius: 12px; padding: 1.25rem; margin: 1rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.25); transition: transform 0.2s, box-shadow 0.2s; }
.card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.35); }
.metric-card { background: linear-gradient(135deg, #161b22 0%, #1c2128 100%); border: 1px solid #30363d; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; }
.badge { display: inline-block; padding: 0.25rem 0.6rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.03em; }
.badge-high { background: rgba(248, 81, 73, 0.2); color: #f85149; border: 1px solid rgba(248, 81, 73, 0.4); }
.badge-med { background: rgba(210, 153, 34, 0.2); color: #d29922; border: 1px solid rgba(210, 153, 34, 0.4); }
.badge-low { background: rgba(63, 185, 80, 0.2); color: #3fb950; border: 1px solid rgba(63, 185, 80, 0.4); }

.alert-high { background: linear-gradient(90deg, rgba(248, 81, 73, 0.15) 0%, transparent 100%); border-left: 4px solid #f85149; padding: 0.85rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0; font-family: 'JetBrains Mono', monospace; }
.alert-medium { background: linear-gradient(90deg, rgba(210, 153, 34, 0.12) 0%, transparent 100%); border-left: 4px solid #d29922; padding: 0.85rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0; }
.alert-banner { background: linear-gradient(135deg, #f85149 0%, #da3633 50%, #b62324 100%); color: white; padding: 1rem 1.5rem; border-radius: 10px; margin-bottom: 1rem; font-weight: 600; box-shadow: 0 4px 16px rgba(248, 81, 73, 0.3); letter-spacing: 0.02em; }
.alert-misinfo { background: linear-gradient(135deg, rgba(210, 153, 34, 0.25) 0%, rgba(158, 106, 3, 0.2) 100%); border: 1px solid rgba(210, 153, 34, 0.5); color: #d29922; padding: 0.9rem 1.2rem; border-radius: 10px; margin: 0.5rem 0; font-weight: 500; }
.stress-high { color: #f85149; }
.stress-med { color: #d29922; }
.stress-low { color: #3fb950; }

div[data-testid="stMetric"] { background: linear-gradient(145deg, #161b22 0%, #21262d 100%); border: 1px solid #30363d; border-radius: 10px; padding: 1rem; }
div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-weight: 500 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #161b22; padding: 6px; border-radius: 12px; border: 1px solid #30363d; }
.stTabs [data-baseweb="tab"] { font-family: 'DM Sans', sans-serif; font-weight: 500; border-radius: 8px; padding: 0.6rem 1rem; }
.stTabs [aria-selected="true"] { background: #30363d !important; }
.sidebar .stSelectbox label { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


def main():
    st.sidebar.header("Controls")
    city = st.sidebar.text_input("City", "London")
    country = st.sidebar.text_input("Country", "UK")
    news_query = st.sidebar.text_input("News search", "technology stocks")
    symbols_str = st.sidebar.text_input("Stocks (comma)", "AAPL,GOOGL,MSFT,AMZN,META")
    symbols = [s.strip() for s in symbols_str.split(",") if s.strip()] or ["AAPL", "GOOGL", "MSFT"]
    refresh_interval = st.sidebar.slider("Auto-refresh (seconds)", 30, 300, 60, 30)
    chart_type = st.sidebar.radio("Finance chart type", ["Bar", "Line", "Area"], horizontal=True)
    show_price_history = st.sidebar.checkbox("Show price history", value=True)

    st_autorefresh(interval=refresh_interval * 1000, key="autorefresh")

    location_key = f"{city}|{country}|{news_query}"
    last_ts = st.session_state.get("last_refresh_ts", 0)
    elapsed = (datetime.now().timestamp() - last_ts) if last_ts else refresh_interval + 1
    needs_refresh = (
        "finance_data" not in st.session_state
        or st.session_state.get("cache_key") != location_key
        or elapsed >= refresh_interval
        or st.sidebar.button("⟳ Refresh Now")
    )
    if needs_refresh:
        with st.spinner("Fetching..."):
            st.session_state.finance_data = {"stocks": fetch_yahoo_finance(symbols), "crypto": fetch_crypto_yahoo()}
            st.session_state.weather_data = fetch_all_weather(city, country)
            st.session_state.cache_key = location_key
            news_df = fetch_all_news([news_query] if news_query else None)
            st.session_state.news_data = add_sentiment_to_news(news_df) if not news_df.empty else news_df
            st.session_state.reddit_data = fetch_reddit_sentiment(limit=10)
            st.session_state.threat_data = fetch_threat_news()
            st.session_state.last_refresh = datetime.now().strftime("%H:%M:%S")
            st.session_state.last_refresh_ts = datetime.now().timestamp()

    st.title("Insight Dashboard")
    if "last_refresh" in st.session_state:
        st.caption(f"Last refresh: {st.session_state.last_refresh}")
    st.divider()

    finance = st.session_state.get("finance_data", {})
    stocks = clean_stock_data(finance.get("stocks", pd.DataFrame()))
    crypto = clean_crypto_data(finance.get("crypto", pd.DataFrame()))
    news = st.session_state.get("news_data", pd.DataFrame())

    alerts = compute_trend_alerts(stocks, crypto)
    crash_alerts = [a for a in alerts if a.get("severity") == "high"]
    misinfo_news = news[news["misinfo_risk"]] if not news.empty and "misinfo_risk" in news.columns and news["misinfo_risk"].any() else pd.DataFrame()

    stock_hist = {s: fetch_stock_history(s) for s in stocks["symbol"].tolist()} if not stocks.empty else {}
    crypto_hist = {s: fetch_crypto_history(s) for s in crypto["symbol"].tolist()} if not crypto.empty else {}
    anomaly_flags = 0
    for sym, prices in stock_hist.items():
        if len(prices) >= 5 and zscore_anomaly(prices)[-1]:
            anomaly_flags += 1
    for sym, prices in crypto_hist.items():
        if len(prices) >= 5 and zscore_anomaly(prices)[-1]:
            anomaly_flags += 1

    news_sent = aggregate_sentiment(news["sentiment"].tolist() if not news.empty and "sentiment" in news.columns else [])["mean"]
    stress = market_stress_score(stocks, crypto, news_sent, anomaly_flags)

    if crash_alerts:
        for a in crash_alerts[:3]:
            st.markdown(f'<div class="alert-banner">◆ {a["message"]}</div>', unsafe_allow_html=True)
    if not misinfo_news.empty:
        for _, r in misinfo_news.head(2).iterrows():
            st.markdown(f'<div class="alert-misinfo">◉ Possible misinformation / deepfake signal: {r.get("title", "")[:80]}...</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "Market Stress", "Finance", "Weather", "News & Sentiment",
        "Risk Tracker", "Deepfake Risk", "Economic Stress", "Threat Intelligence",
        "Predictive Signals", "Predictions",
    ])

    with tab1:
        st.subheader("Market Stress Indicator")
        stress_threshold = st.slider("Alert threshold", 0, 100, 60, key="stress_threshold")
        st.caption("Volatility + sentiment + anomaly signals")
        stress_cls = "stress-high" if stress >= stress_threshold else "stress-med" if stress >= 40 else "stress-low"
        st.metric("Stress Score (0-100)", f"{stress:.0f}", "Above threshold" if stress >= stress_threshold else "Moderate" if stress >= 40 else "Low")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("News Sentiment", f"{news_sent:.2f}", "Negative" if news_sent < -0.2 else "Positive" if news_sent > 0.2 else "Neutral")
        with col2:
            st.metric("Anomaly Flags", anomaly_flags, "Price spikes/drops detected")
        with col3:
            max_ch = max(
                [abs(r.get("change", 0) or 0) for _, r in stocks.iterrows()] +
                [abs(r.get("change_24h", 0) or 0) for _, r in crypto.iterrows()]
            ) if not (stocks.empty and crypto.empty) else 0
            st.metric("Max Move %", f"{max_ch:.1f}%", "Largest single move")
        if not stocks.empty or not crypto.empty:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=stress, domain={"x": [0, 1], "y": [0, 1]}, title={"text": "Stress"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#f85149" if stress >= stress_threshold else "#d29922" if stress >= 40 else "#3fb950"}, "threshold": {"line": {"color": "white"}, "value": stress_threshold}}))
            fig.update_layout(template="plotly_dark", height=250)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if stocks.empty and crypto.empty:
            st.info("No finance data. Try again in a moment.")
        else:
            for a in alerts[:5]:
                cls = "alert-high" if a["severity"] == "high" else "alert-medium"
                st.markdown(f'<div class="{cls}">{a["message"]}</div>', unsafe_allow_html=True)

            focus = st.selectbox("Focus on symbol (price history)", ["None"] + stocks["symbol"].tolist() + crypto["symbol"].tolist(), key="finance_focus")
            if focus != "None" and show_price_history:
                hist = stock_hist.get(focus) or crypto_hist.get(focus)
                if hist and len(hist) >= 2:
                    hist_df = pd.DataFrame({"Day": range(len(hist)), "Price": hist})
                    fig = line_chart(hist_df, "Day", "Price", f"{focus} 30-Day Price History")
                    fig.update_layout(xaxis_title="Trading Day", yaxis_title="Price ($)")
                    st.plotly_chart(fig, use_container_width=True)

            change_threshold = st.slider("Filter by min move %", -20.0, 20.0, -20.0, 1.0, key="change_filter")
            stocks_f = stocks[stocks["change"].abs() >= abs(change_threshold)] if not stocks.empty else stocks
            crypto_f = crypto[crypto["change_24h"].abs() >= abs(change_threshold)] if not crypto.empty else crypto

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Stocks")
                if not stocks_f.empty:
                    if chart_type == "Bar":
                        st.plotly_chart(bar_chart(stocks_f, "symbol", "change", ["#f85149", "#8b949e", "#3fb950"]), use_container_width=True)
                    elif chart_type == "Line":
                        fig = px.line(stocks_f, x="symbol", y="change", markers=True)
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.area(stocks_f, x="symbol", y="change")
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)
                    df = stocks_f[["symbol", "price", "change", "volume"]].copy()
                    df["price"] = df["price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "")
                    df["change"] = df["change"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
                    df["volume"] = df["volume"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
                    st.dataframe(df, use_container_width=True, hide_index=True, column_config={"symbol": st.column_config.TextColumn("Symbol", width="small")})
            with c2:
                st.subheader("Crypto")
                if not crypto_f.empty:
                    if chart_type == "Bar":
                        st.plotly_chart(bar_chart(crypto_f, "symbol", "change_24h", ["#f85149", "#8b949e", "#3fb950"]), use_container_width=True)
                    elif chart_type == "Line":
                        fig = px.line(crypto_f, x="symbol", y="change_24h", markers=True)
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.area(crypto_f, x="symbol", y="change_24h")
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)
                    df = crypto_f[["symbol", "price", "change_24h"]].copy()
                    df["price"] = df["price"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
                    df["change_24h"] = df["change_24h"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
                    st.dataframe(df, use_container_width=True, hide_index=True)

    with tab3:
        w = st.session_state.get("weather_data", {})
        current = w.get("current")
        forecast = w.get("forecast", pd.DataFrame())
        if not current:
            st.warning("Could not load weather for this location")
        else:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader(f"{current.get('city')}, {current.get('country')}")
                st.metric("Temp", f"{current.get('temp', 0):.1f}°C")
                st.metric("Feels like", f"{current.get('feels_like', 0):.1f}°C")
                st.metric("Humidity", f"{current.get('humidity', 0)}%")
            with c2:
                if not forecast.empty:
                    agg = aggregate_forecast_by_day(forecast)
                    if not agg.empty:
                        weather_chart = st.radio("Chart", ["Bar", "Line"], horizontal=True, key="weather_chart")
                        if weather_chart == "Bar":
                            fig = go.Figure()
                            fig.add_trace(go.Bar(x=agg["date"].astype(str), y=agg["temp_max"], name="Max", marker_color="#58a6ff"))
                            fig.add_trace(go.Bar(x=agg["date"].astype(str), y=agg["temp_mean"], name="Mean", marker_color="#8b949e"))
                            fig.add_trace(go.Bar(x=agg["date"].astype(str), y=agg["temp_min"], name="Min", marker_color="#388bfd"))
                            fig.update_layout(barmode="group", template="plotly_dark")
                        else:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=agg["date"].astype(str), y=agg["temp_max"], name="Max", mode="lines+markers", line=dict(color="#58a6ff")))
                            fig.add_trace(go.Scatter(x=agg["date"].astype(str), y=agg["temp_mean"], name="Mean", mode="lines+markers", line=dict(color="#8b949e")))
                            fig.add_trace(go.Scatter(x=agg["date"].astype(str), y=agg["temp_min"], name="Min", mode="lines+markers", line=dict(color="#388bfd")))
                            fig.update_layout(template="plotly_dark", hovermode="x unified")
                        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if news.empty:
            st.warning("Add NEWS_API_KEY to .env")
        else:
            st.subheader("News & Sentiment")
            if "sentiment" in news.columns:
                sent_agg = aggregate_sentiment(news["sentiment"].tolist())
                st.metric("Overall Sentiment", f"{sent_agg['mean']:.2f}", f"Stress: {sent_agg['stress']:.0f}/100")
            sent_min, sent_max = st.slider("Filter by sentiment (-1 to 1)", -1.0, 1.0, (-1.0, 1.0), 0.1, key="sent_filter")
            news_f = news[(news["sentiment"] >= sent_min) & (news["sentiment"] <= sent_max)] if "sentiment" in news.columns else news
            show_misinfo_only = st.checkbox("Show misinformation alerts only", value=False, key="misinfo_filter")
            if show_misinfo_only and "misinfo_risk" in news_f.columns:
                news_f = news_f[news_f["misinfo_risk"] == True]
            search = st.text_input("Search headlines", "", key="news_search", placeholder="Type to filter...")
            if search:
                t = news_f["title"].fillna("").str.contains(search, case=False)
                d = news_f["description"].fillna("").str.contains(search, case=False) if "description" in news_f.columns else t & False
                news_f = news_f[t | d]
            for _, row in news_f.head(15).iterrows():
                sent_badge = f" (sent: {row['sentiment']:.2f})" if "sentiment" in row and pd.notna(row.get("sentiment")) else ""
                misinfo_badge = " [Misinfo risk]" if row.get("misinfo_risk") else ""
                with st.expander(f"**{row.get('title', '')}**{sent_badge}{misinfo_badge} ({row.get('source', '')})"):
                    st.caption(row.get("published_at"))
                    st.write(row.get("description"))
                    if row.get("url"):
                        st.link_button("Read", row["url"])
            reddit = st.session_state.get("reddit_data", pd.DataFrame())
            if not reddit.empty:
                st.subheader("Reddit")
                for _, r in reddit.head(10).iterrows():
                    st.caption(f"r/{r.get('subreddit')} • {r.get('score')} upvotes • {r.get('created')}")
                    st.write(r.get("title", "")[:150])

    with tab5:
        st.subheader("Global Risk Tracker")
        chart_risk = st.radio("Chart type", ["Bar", "Scatter", "Pie"], horizontal=True, key="risk_chart")
        st.caption("News sentiment by topic")
        if not news.empty and "sentiment" in news.columns:
            by_source = news.groupby("source").agg(sentiment=("sentiment", "mean"), count=("title", "count")).reset_index()
            if not by_source.empty:
                if chart_risk == "Bar":
                    fig = px.bar(by_source, x="source", y="sentiment", color="sentiment", color_continuous_scale=["#f85149", "#8b949e", "#3fb950"], color_continuous_midpoint=0)
                elif chart_risk == "Scatter":
                    fig = px.scatter(by_source, x="source", y="sentiment", size="count", color="sentiment", color_continuous_scale=["#f85149", "#3fb950"])
                else:
                    fig = px.pie(by_source, values="count", names="source", color_discrete_sequence=px.colors.sequential.RdBu_r)
                fig.update_layout(template="plotly_dark", title="Sentiment by Source")
                st.plotly_chart(fig, use_container_width=True)
        risk_queries = ["economic recession", "geopolitical", "market crash"]
        st.write("**Risk-related topics:** ", ", ".join(risk_queries))
        st.info("Add more news queries in sidebar to track region/topic-specific sentiment.")

    with tab6:
        st.subheader("AI Misinformation & Deepfake Risk")
        st.caption("Authenticity scoring and deepfake risk signals from news")
        if not news.empty and "deepfake_risk" in news.columns:
            avg_auth = news["authenticity"].mean() if "authenticity" in news.columns else 50
            avg_risk = news["deepfake_risk"].mean()
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Avg Authenticity", f"{avg_auth:.0f}/100", "Higher = more likely authentic")
            with c2:
                st.metric("Avg Deepfake Risk", f"{avg_risk:.0f}/100", "Higher = higher misinfo risk")
            with c3:
                high_risk = (news["deepfake_risk"] > 30).sum() if "deepfake_risk" in news.columns else 0
                st.metric("High-Risk Items", int(high_risk), "Content flagged for review")
            by_source = news.groupby("source").agg(authenticity=("authenticity", "mean"), deepfake_risk=("deepfake_risk", "mean")).reset_index()
            if not by_source.empty:
                fig = px.bar(by_source, x="source", y=["authenticity", "deepfake_risk"], barmode="group", title="Authenticity vs Deepfake Risk by Source")
                fig.update_layout(template="plotly_dark", yaxis_title="Score")
                st.plotly_chart(fig, use_container_width=True)
            st.subheader("Flagged Content (Regional/Global)")
            flagged = news[news["deepfake_risk"] > 20].sort_values("deepfake_risk", ascending=False) if "deepfake_risk" in news.columns else pd.DataFrame()
            for _, r in flagged.head(10).iterrows():
                with st.expander(f"{r.get('source')}: {r.get('title', '')[:60]}... (risk: {r.get('deepfake_risk', 0):.0f})"):
                    st.write(r.get("description", ""))
                    if r.get("url"):
                        st.link_button("Read", r["url"])
        else:
            st.info("News with sentiment data required. Add NEWS_API_KEY to .env.")

    with tab7:
        st.subheader("Economic Stress & Social Sentiment")
        st.caption("Reddit sentiment correlated with market moves")
        reddit = st.session_state.get("reddit_data", pd.DataFrame())
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Market Stress", f"{stress:.0f}/100", "Volatility + sentiment + anomalies")
            max_ch = max([abs(r.get("change", 0) or 0) for _, r in stocks.iterrows()] + [abs(r.get("change_24h", 0) or 0) for _, r in crypto.iterrows()]) if not (stocks.empty and crypto.empty) else 0
            st.metric("Largest Market Move", f"{max_ch:.1f}%", "Today")
        with c2:
            st.metric("News Sentiment", f"{news_sent:.2f}", "Negative" if news_sent < -0.2 else "Positive" if news_sent > 0.2 else "Neutral")
            st.metric("Social Posts (Reddit)", len(reddit), "r/wallstreetbets, r/stocks, r/CryptoCurrency")
        if not reddit.empty:
            st.subheader("Reddit Hot Posts")
            for _, r in reddit.head(8).iterrows():
                st.caption(f"r/{r.get('subreddit')} · {r.get('score')} upvotes")
                st.write(r.get("title", "")[:120])
        st.subheader("Correlation View")
        if not stocks.empty and not news.empty and "sentiment" in news.columns:
            avg_sent = news["sentiment"].mean()
            avg_ch = stocks["change"].mean()
            st.write(f"Avg news sentiment: **{avg_sent:.2f}** | Avg stock change: **{avg_ch:.2f}%**")
            st.info("Negative sentiment often precedes or accompanies market stress. Monitor Reddit + news for early signals.")

    with tab8:
        st.subheader("Cybersecurity Threat Trend Analyzer")
        st.caption("Threat feed integration and NLP classification")
        threat_ip = st.text_input("Check IP (AbuseIPDB)", "", placeholder="e.g. 8.8.8.8", key="threat_ip")
        if threat_ip:
            ip_data = fetch_abuseipdb_reports(ip=threat_ip.strip())
            if not ip_data.empty:
                st.dataframe(ip_data, use_container_width=True, hide_index=True)
            else:
                st.caption("Add ABUSEIPDB_API_KEY to .env for IP lookup. Free at abuseipdb.com")
        threats = st.session_state.get("threat_data", fetch_threat_news())
        if not threats.empty:
            high = threats[threats["threat_level"] == "High"] if "threat_level" in threats.columns else pd.DataFrame()
            st.metric("High-Severity Threats", len(high), "Ransomware, breach, phishing")
            for _, r in threats.head(10).iterrows():
                cats = r.get("threat_cats", []) or []
                level = r.get("threat_level", "Low") or "Low"
                cls = "alert-high" if level == "High" else "alert-medium" if level == "Medium" else ""
                with st.expander(f"[{level}] {r.get('title', '')[:70]}... ({', '.join(cats) if isinstance(cats, list) else cats})"):
                    st.write(r.get("source", ""))
                    if r.get("url"):
                        st.link_button("Read", r["url"])
        if threats.empty:
            st.info("Add NEWS_API_KEY for threat news. Demo data shown when unavailable.")

    with tab9:
        st.subheader("Predictive Signals")
        horizon = st.slider("Prediction horizon (days)", 1, 14, 7, key="pred_horizon")
        summary6 = get_predictions_summary(stocks, crypto, st.session_state.get("weather_data", {}).get("forecast", pd.DataFrame()), stock_hist, crypto_hist)
        from ml_model import predict_trend
        c1, c2 = st.columns(2)
        with c1:
            for s in summary6.get("stocks", []):
                hist = stock_hist.get(s["symbol"], [])
                pred = predict_trend(hist, horizon) if len(hist) >= 5 else None
                st.metric(s["symbol"], f"${s['current']:.2f}", f"{horizon}d: ${pred[-1]:.2f}" if pred else "N/A")
        with c2:
            for c in summary6.get("crypto", []):
                hist = crypto_hist.get(c["symbol"], [])
                pred = predict_trend(hist, horizon) if len(hist) >= 5 else None
                st.metric(c["symbol"], f"${c['current']:,.0f}", f"{horizon}d: ${pred[-1]:,.0f}" if pred else "N/A")
        pred_symbol = st.selectbox("View prediction chart", ["None"] + [s["symbol"] for s in summary6.get("stocks", [])] + [c["symbol"] for c in summary6.get("crypto", [])], key="pred_chart")
        if pred_symbol != "None":
            hist = stock_hist.get(pred_symbol) or crypto_hist.get(pred_symbol)
            pred_vals = predict_trend(hist, horizon) if hist and len(hist) >= 5 else []
            if hist and pred_vals:
                combined = list(hist) + pred_vals
                pred_df = pd.DataFrame({"Day": range(len(combined)), "Price": combined, "Type": ["Actual"] * len(hist) + ["Predicted"] * len(pred_vals)})
                fig = px.line(pred_df, x="Day", y="Price", color="Type", markers=True)
                fig.add_vline(x=len(hist) - 0.5, line_dash="dash", line_color="gray", annotation_text="Now")
                fig.update_layout(template="plotly_dark", title=f"{pred_symbol} Price + {horizon}-Day Prediction")
                st.plotly_chart(fig, use_container_width=True)

    with tab10:
        summary = get_predictions_summary(stocks, crypto, st.session_state.get("weather_data", {}).get("forecast", pd.DataFrame()), stock_hist, crypto_hist)
        st.caption("Linear regression on 30-day history")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Stocks")
            for s in summary.get("stocks", []):
                ch = s.get("change_pct", 0)
                st.metric(s["symbol"], f"${s['current']:.2f}", f"{ch:+.2f}% → ${s.get('predicted', 0):.2f}")
                with st.expander("Details"):
                    st.write(f"**Trend:** {s.get('trend')} • **Volatility:** {s.get('volatility', 0):.1f}%")
        with c2:
            st.subheader("Crypto")
            for c in summary.get("crypto", []):
                ch = c.get("change_pct", 0)
                st.metric(c["symbol"], f"${c['current']:,.2f}", f"{ch:+.2f}% → ${c.get('predicted', 0):,.2f}")
        with c3:
            st.subheader("Weather")
            if summary and summary.get("weather"):
                w = summary["weather"]
                preds = w.get("predicted_temps", [])
                st.write(f"**Trend:** {w.get('trend')}")
                for i, t in enumerate(preds[:5], 1):
                    st.write(f"Day {i}: {t:.1f}°C" if isinstance(t, (int, float)) else f"Day {i}: {t}")

    st.divider()
    st.markdown('<p style="text-align: center; color: #8b949e; font-size: 0.8rem;">Insight Dashboard · Finance · Weather · News · Predictions</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
