# Real-Time Insight Dashboard

**Finance, Weather, News & Risk Signals** – A dashboard that combines live data with sentiment analysis, anomaly detection, and predictive signals for urgent 2026 issues.

---

## Features

| Feature | Description |
|--------|-------------|
| **Market Stress** | Volatility + sentiment + anomaly signals (0–100 score) |
| **Finance** | Stocks & crypto with trend alerts |
| **Weather** | Open-Meteo (no API key) |
| **News & Sentiment** | NLP sentiment, authenticity scoring, misinfo flags |
| **Risk Tracker** | News sentiment by source |
| **Deepfake Risk** | AI misinformation & deepfake detection signals, authenticity scoring |
| **Economic Stress** | Reddit + market + news correlation, stress alerts |
| **Threat Intelligence** | Cybersecurity threat feeds, AbuseIPDB, NLP classification |
| **Predictive Signals** | 7-day market projections |
| **Predictions** | Stock, crypto & weather with volatility |

---

## Tech Stack

- **Python 3.10+**, Pandas, NumPy
- **Streamlit**, Plotly
- **scikit-learn** – Time-series predictions
- **TextBlob** – Sentiment analysis
- **yfinance** – Stocks & crypto
- **Open-Meteo** – Weather (no key)
- **News API** – News
- **PRAW** – Reddit (optional)

---

## Installation

```bash
git clone https://github.com/yourusername/Data-Dashboard.git
cd Data-Dashboard
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add:
- `NEWS_API_KEY` – [newsapi.org](https://newsapi.org/)
- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` – [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
- `ABUSEIPDB_API_KEY` (optional) – [abuseipdb.com](https://www.abuseipdb.com/) for IP threat lookup

## Run

```bash
streamlit run src/dashboard.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## Project Structure

```
Data-Dashboard/
├── src/
│   ├── api_fetcher.py     # Finance, weather, news, Reddit
│   ├── data_processing.py # Clean, sentiment, stress score
│   ├── sentiment.py       # NLP sentiment, authenticity, deepfake risk
│   ├── anomaly.py         # Z-score anomaly detection
│   ├── threat.py          # Threat classification, AbuseIPDB, threat news
│   ├── ml_model.py        # Predictions
│   └── dashboard.py       # Streamlit app
├── requirements.txt
├── .env.example
└── README.md
```

---

## License

MIT
