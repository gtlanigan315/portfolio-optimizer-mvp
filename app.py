import streamlit as st
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns

st.set_page_config(page_title="ETF Portfolio Optimizer", layout="centered")
st.title("\U0001F4C8 Smarter ETF Portfolio Optimizer")

st.markdown("""
Paste your current ETF portfolio and we’ll show you an optimized version—
with better diversification and higher risk-adjusted return.

Example format:  
`SPY: 40, TLT: 40, GLD: 20`
""")

input_text = st.text_area("Your ETF Portfolio", height=100)

if st.button("Optimize Portfolio"):
    try:
        # Parse input
        parts = [x.strip() for x in input_text.split(",") if x.strip()]
        portfolio = {}
        for p in parts:
            ticker, weight = p.split(":")
            portfolio[ticker.strip().upper()] = float(weight.strip()) / 100.0

        tickers = list(portfolio.keys())
        weights = np.array(list(portfolio.values()))

        # Simulated expected returns and covariance
        mock_returns = {
            "SPY": 0.08, "TLT": 0.04, "GLD": 0.06, "TIP": 0.03, "VNQ": 0.07
        }
        mock_cov = pd.DataFrame([
            [0.0225, 0.01,   0.02,   0.01,   0.02],
            [0.01,   0.01,   0.01,   0.01,   0.01],
            [0.02,   0.01,   0.04,   0.01,   0.02],
            [0.01,   0.01,   0.01,   0.0025, 0.01],
            [0.02,   0.01,   0.02,   0.01,   0.0324],
        ], columns=["SPY", "TLT", "GLD", "TIP", "VNQ"], index=["SPY", "TLT", "GLD", "TIP", "VNQ"])

        used = [t for t in tickers if t in mock_returns]
        mu = pd.Series({k: mock_returns[k] for k in used})
        S = mock_cov.loc[used, used]

        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w >= 0)
        ef.add_constraint(lambda w: w <= 0.25)
        ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=False)

        # Results
        st.subheader("\U0001F4CA Optimized Portfolio")
        result_df = pd.DataFrame({
            "ETF": cleaned_weights.keys(),
            "New Weight %": [round(v * 100, 2) for v in cleaned_weights.values()]
        })
        st.dataframe(result_df)

        st.markdown(f"**Expected Annual Return:** {round(perf[0]*100, 2)}%  ")
        st.markdown(f"**Annual Volatility:** {round(perf[1]*100, 2)}%  ")
        st.markdown(f"**Sharpe Ratio:** {round(perf[2], 2)}")

        st.success("Portfolio successfully optimized!")

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()


