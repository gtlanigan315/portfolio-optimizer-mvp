"""
# Smarter ETF Portfolio Optimizer

A Streamlit app that helps ETF investors upgrade their portfolio with institutional logic and macro-aware diversification.

## Features

- **Paste your ETF holdings** (e.g., `SPY: 40, TLT: 40, GLD: 20`)
- **Supports both ETFs and S&P 500 stocks** (uses `etf_metadata.csv` and `sp500_secmaster.csv`)
- **Optimizes portfolio weights** using PyPortfolioOpt, with constraints on max weight per holding
- **Choose investor persona** (Growth, Conservative, Global, Income, 529, or Custom) to set risk preferences
- **Select optimization objective:** Max Sharpe Ratio, Min Volatility, or Max Return
- **Compares your current vs. optimized portfolio** on expected return, volatility, and Sharpe ratio
- **Shows weight breakdown** for each holding before and after optimization
- **10-year growth projection** for both portfolios, visualized as a line chart
- **Handles missing or unsupported tickers** with clear warnings
- **Easy to use:** just paste your portfolio and click "Optimize"

## Run Locally

```bash
pip install -r requirements.txt
streamlit run src/portfolio_optimizer_mvp/app.py
```

## Deploy on Streamlit Cloud

1. Fork this repo to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your repo and click 'Deploy'

## Future enhancements
1. Improve portfolio import- systematic (plaid) or upload pdf statements
2. Broaden assets- crpyto, real estate
3. 
"""

