import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from pypfopt import expected_returns, risk_models, EfficientFrontier

st.set_page_config(page_title="ETF Portfolio Optimizer", layout="centered")
st.title("\U0001F4C8 Smarter ETF Portfolio Optimizer")

st.markdown("""
Paste your current ETF portfolio and we’ll show you an optimized version—
with better diversification and higher risk-adjusted return.

Example format:  
`SPY: 40, TLT: 40, GLD: 20`
""")

input_text = st.text_area("Your ETF Portfolio", height=100)
initial_value = st.number_input("Initial Portfolio Value ($)", min_value=1000, value=100000, step=1000)

# Load reference data
etf_meta = pd.read_csv("csv/etf_metadata.csv")
sp500_meta = pd.read_csv("csv/sp500_secmaster.csv")

# Combine ETF + stock tickers into one universe
etf_universe = etf_meta["Symbol"].tolist()
stock_universe = sp500_meta["Ticker"].tolist()
full_universe = list(set(etf_universe + stock_universe))

# Cache and fetch price data
@st.cache_data(ttl=86400)
def get_price_data(tickers):
    return yf.download(tickers, start="2018-01-01", auto_adjust=True)["Close"].dropna(axis=1)

price_data = get_price_data(full_universe)

# Cache expected returns and covariance matrix
@st.cache_data(ttl=86400)
def get_return_cov_matrix(price_data):
    mu = expected_returns.mean_historical_return(price_data)
    S = risk_models.sample_cov(price_data)
    return mu, S

mu, S = get_return_cov_matrix(price_data)

# Portfolio type selection
persona = st.selectbox("Choose your investor persona", [
    "Growth-Seeking Millennial",
    "Conservative Near-Retiree",
    "Global Diversifier",
    "Income-Focused Retiree",
    "529 College Saver",
    "Custom"
])


# Map portfolio types to volatility floors and default max weights
persona_settings = {
    "Growth-Seeking Millennial": {"target_vol": 0.16, "intl_bias": 0.1, "bond_max": 0.2},
    "Conservative Near-Retiree": {"target_vol": 0.07, "intl_bias": 0.2, "bond_max": 0.6},
    "Global Diversifier": {"target_vol": 0.10, "intl_bias": 0.4, "bond_max": 0.3},
    "Income-Focused Retiree": {"target_vol": 0.06, "intl_bias": 0.1, "bond_max": 0.7},
    "529 College Saver": {"target_vol": 0.08, "intl_bias": 0.2, "bond_max": 0.5},
    "Custom": {}
}


default_max_weight = {
    "Emergency Fund": 1.0,
    "Income": 0.5,
    "Balanced": 0.4,
    "Growth": 0.3,
    "Aggressive": 0.25
}[portfolio_type]

# Optimization preference
objective = st.selectbox("What do you want to optimize for?", ["Max Sharpe Ratio", "Min Volatility", "Max Return"])
max_weight = st.slider("Maximum weight per ETF (%)", min_value=5, max_value=100, value=int(default_max_weight * 100), step=5) / 100.0

if st.button("Optimize Portfolio"):
    try:
        # Parse user input
        parts = [x.strip() for x in input_text.split(",") if x.strip()]
        portfolio = {}
        for p in parts:
            ticker, weight = p.split(":")
            portfolio[ticker.strip().upper()] = float(weight.strip()) / 100.0

        tickers = list(portfolio.keys())
        available = list(mu.index)
        used = [t for t in tickers if t in available]
        dropped = [t for t in tickers if t not in used]
        if dropped:
            st.warning(f"The following tickers are not in the dataset and were ignored: {', '.join(dropped)}")
        st.write("Portfolio tickers:", tickers)
        st.write("Tickers in data:", available)
        st.write("Valid tickers used:", used)
        # Calculate weights for user's current portfolio for comparison
        weights = np.array([portfolio[t] for t in used])

        used = [t for t in tickers if t in mu.index]
        if len(used) < 2:
            st.warning("Please enter at least 2 supported ETFs from this list: " + ", ".join(mu.index))
            st.stop()

        # Optimization from full ETF universe
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w >= 0)
        ef.add_constraint(lambda w: w <= max_weight)

        try:
            if objective == "Max Sharpe Ratio":
                ef.max_sharpe()
            elif objective == "Min Volatility":
                ef.min_volatility()
            elif objective == "Max Return":
                ef.max_quadratic_utility()
        except Exception as e:
            st.error(f"Primary solver failed: {e}. Retrying with SCS...")
            ef = EfficientFrontier(mu, S, solver="SCS")
            ef.add_constraint(lambda w: w >= 0)
            ef.add_constraint(lambda w: w <= max_weight)
            if objective == "Max Sharpe Ratio":
                ef.max_sharpe()
            elif objective == "Min Volatility":
                ef.min_volatility()
            elif objective == "Max Return":
                ef.max_quadratic_utility()

        cleaned_weights = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=False)
        cleaned_weights = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=False)

        # Display performance metrics
        st.subheader("\U0001F4C8 Portfolio Comparison")

        current_return = sum([portfolio[t] * mu[t] for t in used])
        current_vol = np.sqrt(np.dot(weights.T, np.dot(S.loc[used, used].values, weights)))
        current_sharpe = (current_return - 0.02) / current_vol

        perf_df = pd.DataFrame({
            "Metric": ["Expected Return", "Volatility", "Sharpe Ratio"],
            "Current Portfolio": [f"{current_return*100:.2f}%", f"{current_vol*100:.2f}%", f"{current_sharpe:.2f}"],
            "Optimized Portfolio": [f"{perf[0]*100:.2f}%", f"{perf[1]*100:.2f}%", f"{perf[2]:.2f}"]
        })

        st.dataframe(perf_df)

        # Weight comparison
        all_etfs = list(cleaned_weights.keys())
        weight_table = pd.DataFrame({
            "ETF": all_etfs,
            "Current Weight %": [round(portfolio.get(t, 0) * 100, 2) for t in all_etfs],
            "Optimized Weight %": [round(cleaned_weights.get(t, 0) * 100, 2) for t in all_etfs]
        })

        st.subheader("📊 Portfolio Weights (Full Comparison)")
        st.dataframe(weight_table)

        # Time series projection
        st.subheader("\U0001F4C8 10-Year Growth Projection")
        current_year = datetime.datetime.now().year
        years = [str(y) for y in range(current_year, current_year + 11)]
        opt_vals = [initial_value * ((1 + perf[0]) ** i) for i in range(11)]
        cur_vals = [initial_value * ((1 + current_return) ** i) for i in range(11)]

        projection_df = pd.DataFrame({
            "Year": years,
            "Current Portfolio": cur_vals,
            "Optimized Portfolio": opt_vals
        })

        st.line_chart(projection_df.set_index("Year"))

        st.success("Portfolio successfully optimized!")

    except Exception as e:
        st.error(f"Optimization failed: {e}")
        st.stop()
