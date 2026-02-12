# 📊 Real-Time Multi-API Data Dashboard

**Track Finance, Weather, and Trends Live** – A Python-based dashboard that fetches live data from multiple APIs, visualizes trends, and provides actionable insights with optional predictive analytics.

---

## Problem Statement

Accessing multiple real-time data streams and analyzing trends manually is inefficient. This dashboard consolidates key information—finance, weather, news, and predictions—in one interactive, visual interface.

## Solution

A Python-based dashboard that:
- Fetches live data from **Finance** (Yahoo), **Weather** (OpenWeatherMap), and **News** (News API)
- Visualizes trends with interactive Plotly charts
- Highlights unusual moves with **trend alerts**
- Provides **short-term predictions** via simple ML models (optional)
- Auto-refreshes on a configurable schedule

---

## Features

| Feature | Description |
|--------|-------------|
| **Multi-source aggregation** | Stocks, crypto, weather, and news in one place |
| **Interactive graphs** | Bar charts, trend lines, and forecast visualizations |
| **Trend alerts** | Highlights unusual price movements (e.g., >5% moves) |
| **Short-term predictions** | Regression / moving-average predictions for demo |
| **User-friendly UI** | Tabs, dropdowns, sidebar controls, dark theme |
| **Real-time updates** | Manual refresh + optional auto-refresh |

---

## Tech Stack & Libraries

- **Python 3.10+**
- **Pandas** – Data processing & cleaning
- **Streamlit** – Dashboard UI
- **Plotly** – Interactive visualizations
- **scikit-learn** – Optional ML predictions
- **yfinance** – Stock & crypto (no API key)
- **requests** – HTTP for OpenWeatherMap & News API

---

## Installation & Usage

### 1. Clone & set up

```bash
git clone https://github.com/yourusername/real-time-dashboard.git
cd real-time-dashboard
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys (optional)

- Copy `.env.example` to `.env`
- Add your keys (see links below)

| API | Get key | Required for |
|-----|---------|--------------|
| **Yahoo Finance** | Built-in (no key) | Stocks & crypto |
| **OpenWeatherMap** | [openweathermap.org/api](https://openweathermap.org/api) | Weather tab |
| **News API** | [newsapi.org](https://newsapi.org/) | News tab |
| **Alpha Vantage** | [alphavantage.co](https://www.alphavantage.co/support/#api-key) | Extra finance data |

### 5. Run the dashboard

```bash
streamlit run src/dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
real-time-dashboard/
├── data/                  # Sample datasets (optional)
├── src/
│   ├── api_fetcher.py     # Fetch data from APIs
│   ├── data_processing.py # Clean & process data
│   ├── dashboard.py       # Streamlit app
│   ├── ml_model.py        # Optional predictive model
├── assets/                # Images, diagrams, sample outputs
├── requirements.txt
├── .env.example
└── README.md
```

---

## Sample Outputs

- **Finance tab**: Stock/crypto prices, change %, volume, alerts
- **Weather tab**: Current conditions + 5-day forecast
- **News tab**: Headlines from News API
- **Predictions tab**: Short-term stock/weather predictions

Add screenshots to `assets/` and link them here for your portfolio.

---

## Deployment (Streamlit Cloud)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `src/dashboard.py` as main file
4. Add **Secrets** (API keys) in Streamlit Cloud dashboard

---

## Future Improvements

- [ ] Add more APIs (Reddit, Twitter/X, Alpha Vantage)
- [ ] Enhance ML predictions (LSTM, Prophet)
- [ ] PySpark for large-scale data processing
- [ ] Deploy as public web app with auth

---

## License

MIT
