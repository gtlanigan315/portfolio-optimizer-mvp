import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from pypfopt import expected_returns, risk_models, EfficientFrontier
import os
from dotenv import load_dotenv
import plotly.graph_objs as go
import logging

# Add Gemini for chat
import google.generativeai as genai

load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("portfolio_optimizer.log"),
        logging.StreamHandler()
    ]
)

st.set_page_config(page_title="ETF Portfolio Optimizer", layout="centered")
st.title("\U0001F4C8 Smarter ETF Portfolio Optimizer")

st.markdown("""
Paste your current ETF portfolio and we’ll show you an optimized version—
with better diversification and higher risk-adjusted return.

Example format:  
`SPY: 40000, TLT: 40000, GLD: 20000`
""")

# --- Table Editor ---
st.markdown("Use the table editor below or upload a CSV to enter your portfolio:")
if "portfolio_table" not in st.session_state:
    st.session_state["portfolio_table"] = pd.DataFrame({
        "Ticker": ["SPY", "TLT", "GLD"],
        "Dollars": [40000, 40000, 20000]
    })

edited_df = st.data_editor(
    st.session_state["portfolio_table"],
    num_rows="dynamic",
    use_container_width=True,
    key="portfolio_editor"
)

# --- CSV Upload Option ---
uploaded_file = st.file_uploader("Or upload a CSV with columns: Ticker, Dollars", type=["csv"])
if uploaded_file:
    df_upload = pd.read_csv(uploaded_file)
    st.session_state["portfolio_table"] = df_upload
    st.success("Portfolio loaded from CSV!")

# Always use the table as the source of truth
portfolio_df = st.session_state["portfolio_table"]

# Load reference data
etf_path = "src/portfolio_optimizer_mvp/csv/etf_metadata.csv"
sp500_path = "src/portfolio_optimizer_mvp/csv/sp500_secmaster.csv"

if not os.path.exists(etf_path) or not os.path.exists(sp500_path):
    st.error(f"Required data files not found. Please ensure '{etf_path}' and '{sp500_path}' exist.")
    st.stop()
#TODO: figure out why there is an error loading the etf metdata csv
etf_meta = pd.read_csv(etf_path, on_bad_lines="skip")
# Clean up column names for robustness
etf_meta.columns = etf_meta.columns.str.strip().str.replace(" ", "").str.lower()

# Only use ETF tickers for now
etf_universe = etf_meta["symbol"].tolist()
full_universe = etf_universe  # Only ETFs

# Identify international and bond ETFs using new columns
intl_etfs = etf_meta[etf_meta["region"] == "International"]["symbol"].tolist()
bond_etfs = etf_meta[etf_meta["category"] == "Bond"]["symbol"].tolist()

# Identify US equity ETFs (for possible future constraints)
us_equity_etfs = etf_meta[(etf_meta["region"] == "US") & (etf_meta["category"] == "Equity")]["symbol"].tolist()

# Identify sector ETFs (for possible sector constraints)
# Example: all technology sector equity ETFs
tech_etfs = etf_meta[(etf_meta["category"] == "Equity") & (etf_meta["sector"] == "Technology")]["symbol"].tolist()

# Cache and fetch price data
@st.cache_data(ttl=86400)
def get_price_data(tickers):
    return yf.download(tickers, start="2018-01-01", auto_adjust=True)["Close"].dropna(axis=1)

price_data = get_price_data(full_universe)
price_data = price_data.dropna(axis=1, thresh=int(0.95 * len(price_data)))

# Cache expected returns and covariance matrix
@st.cache_data(ttl=86400)
def get_return_cov_matrix(price_data):
    mu = expected_returns.mean_historical_return(price_data)
    S = risk_models.CovarianceShrinkage(price_data).ledoit_wolf()
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

# --- New Section: Adjustable Optimization Parameters ---
st.header("Adjust Optimization Parameters")

# Get persona defaults
defaults = persona_settings.get(persona, {})
default_vol = defaults.get("target_vol", 0.10)
default_intl = defaults.get("intl_bias", 0.1)
default_bond = defaults.get("bond_max", 0.3)
default_max_weight = 0.3

# Sliders for optimization parameters, initialized to persona defaults
target_vol = st.slider(
    "Target Volatility (annualized, %)", 
    min_value=3.0, max_value=25.0, value=float(default_vol * 100), step=0.1
) / 100.0

intl_bias = st.slider(
    "Minimum International Allocation (%)", 
    min_value=0.0, max_value=100.0, value=float(default_intl * 100), step=1.0
) / 100.0

bond_max = st.slider(
    "Maximum Bond Allocation (%)", 
    min_value=0.0, max_value=100.0, value=float(default_bond * 100), step=1.0
) / 100.0

max_weight = st.slider(
    "Maximum weight per ETF (%)", 
    min_value=5, max_value=100, value=int(default_max_weight * 100), step=5
) / 100.0

# Optimization preference
objective = st.selectbox(
    "What do you want to optimize for?", 
    ["Max Sharpe Ratio", "Min Volatility", "Max Return"]
)

# Baseline portfolio options for efficient frontier
baseline_options = {
    "60/40 (SPY/AGG)": {"SPY": 0.6, "AGG": 0.4},
    "Current Portfolio": "current",  # special handling
    "100% SPY": {"SPY": 1.0},
    "100% AGG": {"AGG": 1.0},
}
default_baselines = ["60/40 (SPY/AGG)", "Current Portfolio", "100% SPY"]
selected_baselines = st.multiselect(
    "Show baseline portfolios on Efficient Frontier:",
    options=list(baseline_options.keys()),
    default=default_baselines
)

# --- Debug Mode Option ---
debug_mode = st.sidebar.checkbox("Debug Mode", value=False, help="Show optimization diagnostics and constraint violations.")

if st.button("Optimize Portfolio"):
    try:
        # Build portfolio dict from table
        portfolio = {}
        total_dollars = 0
        for _, row in portfolio_df.iterrows():
            ticker = str(row["Ticker"]).strip().upper()
            try:
                dollars = float(row["Dollars"])
            except Exception:
                continue
            if ticker and dollars > 0:
                portfolio[ticker] = dollars
                total_dollars += dollars

        if total_dollars == 0:
            st.warning("Total portfolio value must be greater than zero.")
            st.stop()

        # Convert dollar weights to fractions
        portfolio_weights = {t: v / total_dollars for t, v in portfolio.items()}

        tickers = list(portfolio.keys())
        available = list(mu.index)
        # Only allow ETFs (intersection with ETF universe)
        used = [t for t in tickers if t in etf_universe and t in available]
        dropped = [t for t in tickers if t not in used]
        if dropped:
            st.warning(f"The following tickers are not in the ETF dataset and were ignored: {', '.join(dropped)}")
        weights = np.array([portfolio_weights[t] for t in used])

        if len(used) < 2:
            st.warning("Please enter at least 2 supported ETFs from this list: " + ", ".join(etf_universe))
            st.stop()

        # Optimization from full ETF universe
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w >= 0)
        ef.add_constraint(lambda w: w <= max_weight)

        # Enforce minimum international equity allocation
        if intl_bias > 0 and len(intl_etfs) > 0:
            intl_indices = [i for i, t in enumerate(mu.index) if t in intl_etfs]
            if intl_indices:
                ef.add_constraint(lambda w: sum([w[i] for i in intl_indices]) >= intl_bias)

        # Enforce maximum bond allocation
        if bond_max < 1.0 and len(bond_etfs) > 0:
            bond_indices = [i for i, t in enumerate(mu.index) if t in bond_etfs]
            if bond_indices:
                ef.add_constraint(lambda w: sum([w[i] for i in bond_indices]) <= bond_max)

        # (Optional) Example: Enforce max allocation to a single sector (e.g., Technology)
        # sector_max = 0.4
        # tech_indices = [i for i, t in enumerate(mu.index) if t in tech_etfs]
        # if tech_indices:
        #     ef.add_constraint(lambda w: sum([w[i] for i in tech_indices]) <= sector_max)

        try:
            if objective == "Max Sharpe Ratio":
                ef.max_sharpe()
            elif objective == "Min Volatility":
                ef.min_volatility()
            elif objective == "Max Return":
                ef.max_quadratic_utility()
            opt_status = ef._solver_status if hasattr(ef, "_solver_status") else "unknown"
            opt_diagnostics = ef._solver_output if hasattr(ef, "_solver_output") else None
        except Exception as e:
            logging.exception("Primary solver failed, Error: %s", e)
            st.error(f"Primary solver failed: {e}. Retrying with SCS...")
            ef = EfficientFrontier(mu, S, solver="SCS")
            ef.add_constraint(lambda w: w >= 0)
            ef.add_constraint(lambda w: w <= max_weight)
            try:
                if objective == "Max Sharpe Ratio":
                    ef.max_sharpe()
                elif objective == "Min Volatility":
                    ef.min_volatility()
                elif objective == "Max Return":
                    ef.max_quadratic_utility()
                opt_status = ef._solver_status if hasattr(ef, "_solver_status") else "unknown"
                opt_diagnostics = ef._solver_output if hasattr(ef, "_solver_output") else None
            except Exception as e2:
                logging.exception("SCS solver also failed, Error: %s", e2)
                st.error(f"SCS solver also failed: {e2}")
                if debug_mode:
                    st.error(f"Exception details: {e2}")
                raise

        # --- Debug/Diagnostics Output ---
        if debug_mode:
            st.subheader("Optimization Diagnostics")
            st.write(f"Solver Status: {opt_status}")
            if opt_diagnostics:
                st.code(str(opt_diagnostics))
            # Constraint violation checks
            st.subheader("Constraint Checks")
            cleaned_weights = ef.clean_weights()
            weights_arr = np.array([cleaned_weights[t] for t in mu.index])
            violations = []
            # Max weight per ETF
            if np.any(weights_arr > max_weight + 1e-6):
                violations.append(f"Some ETF weights exceed the max_weight constraint ({max_weight*100:.1f}%).")
            # Min international
            if intl_bias > 0 and len(intl_etfs) > 0:
                intl_indices = [i for i, t in enumerate(mu.index) if t in intl_etfs]
                intl_sum = weights_arr[intl_indices].sum() if intl_indices else 0
                if intl_sum + 1e-6 < intl_bias:
                    violations.append(f"International allocation is below the minimum ({intl_sum:.2%} < {intl_bias:.2%}).")
            # Max bond
            if bond_max < 1.0 and len(bond_etfs) > 0:
                bond_indices = [i for i, t in enumerate(mu.index) if t in bond_etfs]
                bond_sum = weights_arr[bond_indices].sum() if bond_indices else 0
                if bond_sum - 1e-6 > bond_max:
                    violations.append(f"Bond allocation exceeds the maximum ({bond_sum:.2%} > {bond_max:.2%}).")
            if violations:
                for v in violations:
                    st.error(f"Constraint violation: {v}")
            else:
                st.success("No constraint violations detected.")

        cleaned_weights = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=False)

        # Calculate current portfolio stats
        current_return = sum([portfolio_weights[t] * mu[t] for t in used])
        current_vol = np.sqrt(np.dot(weights.T, np.dot(S.loc[used, used].values, weights)))
        current_sharpe = (current_return - 0.02) / current_vol

        # Calculate additional stats BEFORE summary section
        def calc_stats(weights, tickers):
            port_return = float(np.dot(weights, mu.loc[tickers]))
            port_vol = float(np.sqrt(np.dot(weights.T, np.dot(S.loc[tickers, tickers], weights))))
            port_sharpe = (port_return - 0.02) / port_vol
            port_prices = (price_data[tickers] * weights).sum(axis=1)
            running_max = np.maximum.accumulate(port_prices)
            drawdown = (port_prices - running_max) / running_max
            max_drawdown = drawdown.min()
            rolling = port_prices.pct_change(252)
            worst_year = rolling.min()
            best_year = rolling.max()
            return {
                "Expected Return": port_return,
                "Volatility": port_vol,
                "Sharpe Ratio": port_sharpe,
                "Max Drawdown": max_drawdown,
                "Worst 1-Year Return": worst_year,
                "Best 1-Year Return": best_year,
            }

        curr_stats = calc_stats(weights, used)
        all_etfs = list(cleaned_weights.keys())
        opt_weights = np.array([cleaned_weights[t] for t in all_etfs])
        opt_tickers = all_etfs
        opt_stats = calc_stats(opt_weights, opt_tickers)

        # --- 1. Summary and Recommendation ---
        st.header("1. Summary and Recommendation")

        # Compose a prompt for Gemini
        import google.generativeai as genai

        # Prepare a summary of changes
        changes = []
        for t in set(list(portfolio_weights.keys()) + list(cleaned_weights.keys())):
            old = portfolio_weights.get(t, 0)
            new = cleaned_weights.get(t, 0)
            if abs(old - new) > 0.01:
                changes.append(f"{t}: {old*100:.1f}% → {new*100:.1f}%")
        changes_str = "; ".join(changes) if changes else "No significant weight changes."

        perf_improvement = perf[2] - current_sharpe
        prompt = (
            f"You are a financial assistant. Summarize the key changes and benefits of the optimized portfolio for a user. "
            f"Here are the portfolio changes: {changes_str}. "
            f"The current portfolio Sharpe ratio is {current_sharpe:.2f}, and the optimized portfolio Sharpe ratio is {perf[2]:.2f}. "
            f"Expected return changed from {current_return*100:.2f}% to {perf[0]*100:.2f}%. "
            f"Volatility changed from {current_vol*100:.2f}% to {perf[1]*100:.2f}%. "
            f"Max drawdown improved from {curr_stats['Max Drawdown']*100:.2f}% to {opt_stats['Max Drawdown']*100:.2f}%. "
            f"Summarize the main benefits and any tradeoffs in 2-4 sentences."
        )

        # Call Gemini API (make sure you have set your API key in your environment)
        try:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            summary = response.text.strip()
            st.info(summary)
        except Exception as e:
            logging.exception("Gemini API call failed, falling back to simple logic, Error: %s", e)
            # Fallback to simple logic if Gemini fails
            if perf_improvement > 0.05:
                st.success(f"**Recommendation:** Your optimized portfolio has a significantly higher Sharpe ratio ({perf[2]:.2f}) than your current portfolio ({current_sharpe:.2f}). Consider rebalancing for better risk-adjusted returns.")
            elif perf_improvement > 0.01:
                st.info(f"**Recommendation:** The optimized portfolio offers a modest improvement in Sharpe ratio ({perf[2]:.2f} vs {current_sharpe:.2f}).")
            else:
                st.warning(f"**Recommendation:** Your current portfolio is already close to optimal. Only minor improvements are possible.")

        # --- 2. Portfolio Comparison and Summary Stats ---
        st.header("2. Portfolio Comparison and Summary Stats")
        
        # Calculate additional stats
        def calc_stats(weights, tickers):
            # Annualized stats
            port_return = float(np.dot(weights, mu.loc[tickers]))
            port_vol = float(np.sqrt(np.dot(weights.T, np.dot(S.loc[tickers, tickers], weights))))
            port_sharpe = (port_return - 0.02) / port_vol
            # Max drawdown (approximate, based on price_data)
            port_prices = (price_data[tickers] * weights).sum(axis=1)
            running_max = np.maximum.accumulate(port_prices)
            drawdown = (port_prices - running_max) / running_max
            max_drawdown = drawdown.min()
            # Worst 1-year return (rolling 252 days)
            rolling = port_prices.pct_change(252)
            worst_year = rolling.min()
            # Best 1-year return
            best_year = rolling.max()
            return {
                "Expected Return": port_return,
                "Volatility": port_vol,
                "Sharpe Ratio": port_sharpe,
                "Max Drawdown": max_drawdown,
                "Worst 1-Year Return": worst_year,
                "Best 1-Year Return": best_year,
            }

        curr_stats = calc_stats(weights, used)
        opt_weights = np.array([cleaned_weights[t] for t in all_etfs])
        opt_tickers = all_etfs
        opt_stats = calc_stats(opt_weights, opt_tickers)

        perf_df = pd.DataFrame({
            "Metric": [
                "Expected Return",
                "Volatility",
                "Sharpe Ratio",
                "Max Drawdown",
                "Worst 1-Year Return",
                "Best 1-Year Return"
            ],
            "Current Portfolio": [
                f"{curr_stats['Expected Return']*100:.2f}%",
                f"{curr_stats['Volatility']*100:.2f}%",
                f"{curr_stats['Sharpe Ratio']:.2f}",
                f"{curr_stats['Max Drawdown']*100:.2f}%",
                f"{curr_stats['Worst 1-Year Return']*100:.2f}%",
                f"{curr_stats['Best 1-Year Return']*100:.2f}%"
            ],
            "Optimized Portfolio": [
                f"{opt_stats['Expected Return']*100:.2f}%",
                f"{opt_stats['Volatility']*100:.2f}%",
                f"{opt_stats['Sharpe Ratio']:.2f}",
                f"{opt_stats['Max Drawdown']*100:.2f}%",
                f"{opt_stats['Worst 1-Year Return']*100:.2f}%",
                f"{opt_stats['Best 1-Year Return']*100:.2f}%"
            ]
        })
        st.dataframe(perf_df)

        # Weight comparison
        all_etfs = list(cleaned_weights.keys())
        weight_table = pd.DataFrame({
            "ETF": all_etfs,
            "Current $": [round(portfolio.get(t, 0), 2) for t in all_etfs],
            "Current Weight %": [round(portfolio_weights.get(t, 0) * 100, 2) for t in all_etfs],
            "Optimized Weight %": [round(cleaned_weights.get(t, 0) * 100, 2) for t in all_etfs],
            "Optimized $": [round(cleaned_weights.get(t, 0) * total_dollars, 2) for t in all_etfs]
        })
        weight_table = weight_table[
            (weight_table["Current $"] != 0) | (weight_table["Optimized $"] != 0)
        ]
        st.subheader("Portfolio Weights (Full Comparison)")
        st.dataframe(weight_table)

        # --- 3. Projections ---
        st.header("3. Projections")
        st.subheader("10-Year Growth Projection")
        current_year = datetime.datetime.now().year
        years = [str(y) for y in range(current_year, current_year + 11)]
        opt_vals = [total_dollars * ((1 + perf[0]) ** i) for i in range(11)]
        cur_vals = [total_dollars * ((1 + current_return) ** i) for i in range(11)]
        projection_df = pd.DataFrame({
            "Year": years,
            "Current Portfolio": cur_vals,
            "Optimized Portfolio": opt_vals
        })
        st.line_chart(projection_df.set_index("Year"))

        # --- 4. Efficient Frontier ---
        st.header("4. Efficient Frontier")

        # Prepare data for plotting
        def get_portfolio_perf(weights_dict):
            tickers = [t for t in weights_dict if t in mu.index]
            weights = np.array([weights_dict[t] for t in tickers])
            port_return = sum([weights_dict[t] * mu[t] for t in tickers])
            port_vol = np.sqrt(np.dot(weights.T, np.dot(S.loc[tickers, tickers].values, weights)))
            port_sharpe = (port_return - 0.02) / port_vol
            return port_return, port_vol, port_sharpe

        # Efficient frontier curve
        frontier_returns = np.linspace(0.03, 0.18, 40)
        frontier_vols = []
        for r in frontier_returns:
            ef_tmp = EfficientFrontier(mu, S)
            ef_tmp.add_constraint(lambda w: w >= 0)
            ef_tmp.add_constraint(lambda w: w <= max_weight)
            try:
                ef_tmp.efficient_return(target_return=r)
                w = ef_tmp.clean_weights()
                tickers = [t for t in w if w[t] > 1e-4]
                weights = np.array([w[t] for t in tickers])
                port_vol = np.sqrt(np.dot(weights.T, np.dot(S.loc[tickers, tickers].values, weights)))
                frontier_vols.append(port_vol)
            except:
                frontier_vols.append(np.nan)

        # Plotly chart
        fig = go.Figure()

        # Plot efficient frontier
        fig.add_trace(go.Scatter(
            x=frontier_vols, y=frontier_returns,
            mode="lines", name="Efficient Frontier",
            line=dict(color="royalblue", width=3)
        ))

        # Plot current portfolio
        fig.add_trace(go.Scatter(
            x=[current_vol], y=[current_return],
            mode="markers", name="Current Portfolio",
            marker=dict(color="red", size=12, symbol="x"),
            text=["Current Portfolio"]
        ))

        # Plot optimized portfolio
        fig.add_trace(go.Scatter(
            x=[perf[1]], y=[perf[0]],
            mode="markers", name="Optimized Portfolio",
            marker=dict(color="green", size=12, symbol="circle"),
            text=["Optimized Portfolio"]
        ))

        # Plot selected baselines
        for baseline in selected_baselines:
            if baseline == "Current Portfolio":
                # Already plotted above, skip or highlight if you want
                continue
            weights_dict = baseline_options[baseline]
            b_ret, b_vol, b_sharpe = get_portfolio_perf(weights_dict)
            fig.add_trace(go.Scatter(
                x=[b_vol], y=[b_ret],
                mode="markers", name=f"{baseline}",
                marker=dict(size=12, symbol="diamond"),
                text=[f"{baseline}"]
            ))

        fig.update_layout(
            title="Efficient Frontier (Risk vs. Return)",
            xaxis_title="Volatility (Std Dev, Annualized)",
            yaxis_title="Expected Return (Annualized)",
            legend=dict(x=0.01, y=0.99),
            height=500,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True, key="main_fig")

        st.success("Portfolio successfully optimized!")

        # After successful optimization, store everything needed for display
        st.session_state["last_optimization"] = True
        st.session_state["last_stats"] = {
            "current_sharpe": f"{current_sharpe:.2f}",
            "optimized_sharpe": f"{perf[2]:.2f}",
            "current_return": f"{current_return*100:.2f}%",
            "optimized_return": f"{perf[0]*100:.2f}%",
            "current_vol": f"{current_vol*100:.2f}%",
            "optimized_vol": f"{perf[1]*100:.2f}%"
        }
        st.session_state["last_summary"] = summary  # from Gemini or fallback
        st.session_state["last_perf_df"] = perf_df
        st.session_state["last_weight_table"] = weight_table
        st.session_state["last_projection_df"] = projection_df
        st.session_state["last_fig"] = fig

        # Also store optimization params for chat context
        st.session_state["persona"] = persona
        st.session_state["target_vol"] = target_vol
        st.session_state["intl_bias"] = intl_bias
        st.session_state["bond_max"] = bond_max
        st.session_state["max_weight"] = max_weight
        st.session_state["objective"] = objective

    except Exception as e:
        logging.exception("Portfolio optimization failed, Error:%s", e)
        st.error(f"Optimization failed: {e}")
        st.stop()

# --- Results Display Section ---
# Show results if available
if st.session_state.get("last_optimization", False):
    st.header("1. Summary and Recommendation")
    st.info(st.session_state.get("last_summary", ""))
    
    st.header("2. Portfolio Comparison and Summary Stats")
    st.dataframe(st.session_state.get("last_perf_df", pd.DataFrame()))
    st.subheader("Portfolio Weights (Full Comparison)")
    st.dataframe(st.session_state.get("last_weight_table", pd.DataFrame()))
    
    st.header("3. Projections")
    st.subheader("10-Year Growth Projection")
    st.line_chart(st.session_state.get("last_projection_df", pd.DataFrame()).set_index("Year"))
    
    st.header("4. Efficient Frontier")
    st.plotly_chart(st.session_state.get("last_fig"), use_container_width=True, key="results_fig")

# --- AI Chatbot Section (place this at the very end of your script) ---
if st.session_state.get("last_optimization", False):
    with st.sidebar.expander("💬 AI Portfolio Chatbot", expanded=False):
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        st.markdown("**Ask anything about your portfolio, optimization, or ETFs!**")

        # Use a form to isolate the chat input from the rest of the app
        with st.form("ai_chat_form", clear_on_submit=True):
            user_input = st.text_input("Type your question:", key="chat_input")
            submitted = st.form_submit_button("Ask AI")
            if submitted and user_input.strip():
                # Compose context from portfolio
                portfolio_context = ""
                if "portfolio_table" in st.session_state:
                    df = st.session_state["portfolio_table"]
                    portfolio_context = (
                        "The user's current ETF portfolio is:\n" +
                        "\n".join([f"{row['Ticker']}: ${row['Dollars']}" for _, row in df.iterrows()])
                    )
                else:
                    portfolio_context = "The user has not entered a portfolio yet."

                # Add optimization params if available
                persona = st.session_state.get("persona", "Custom")
                target_vol = st.session_state.get("target_vol", None)
                intl_bias = st.session_state.get("intl_bias", None)
                bond_max = st.session_state.get("bond_max", None)
                max_weight = st.session_state.get("max_weight", None)
                objective = st.session_state.get("objective", None)

                # Add latest stats if available
                stats_context = ""
                if "last_stats" in st.session_state:
                    stats = st.session_state["last_stats"]
                    stats_context = (
                        f"\nCurrent Portfolio Sharpe Ratio: {stats.get('current_sharpe', 'N/A')}\n"
                        f"Optimized Portfolio Sharpe Ratio: {stats.get('optimized_sharpe', 'N/A')}\n"
                        f"Current Portfolio Expected Return: {stats.get('current_return', 'N/A')}\n"
                        f"Optimized Portfolio Expected Return: {stats.get('optimized_return', 'N/A')}\n"
                        f"Current Portfolio Volatility: {stats.get('current_vol', 'N/A')}\n"
                        f"Optimized Portfolio Volatility: {stats.get('optimized_vol', 'N/A')}\n"
                    )

                params_context = (
                    f"Optimization persona: {persona}\n"
                    f"Target volatility: {target_vol}\n"
                    f"International allocation min: {intl_bias}\n"
                    f"Bond allocation max: {bond_max}\n"
                    f"Max weight per ETF: {max_weight}\n"
                    f"Objective: {objective}\n"
                )

                prompt = (
                    f"You are a helpful financial assistant. "
                    f"{portfolio_context}\n"
                    f"Optimization parameters:\n{params_context}\n"
                    f"{stats_context}\n"
                    f"User question: {user_input}\n"
                    f"Answer in a concise, friendly, and clear way."
                )

                try:
                    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                    model = genai.GenerativeModel("gemini-2.0-flash")
                    response = model.generate_content(prompt)
                    ai_answer = response.text.strip()
                except Exception as e:
                    ai_answer = f"Sorry, I couldn't get an answer from the AI right now. ({e})"
                    logging.exception("Gemini chatbot failed: %s", e)

                st.session_state["chat_history"].append(("user", user_input))
                st.session_state["chat_history"].append(("ai", ai_answer))

        # Display chat history
        for speaker, msg in st.session_state["chat_history"]:
            if speaker == "user":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**AI:** {msg}")

# --- Backtest: Historical Performance ---
if st.session_state.get("last_optimization", False):
    # Use weights and price_data from session or previous optimization
    used = [t for t in portfolio_weights if t in price_data.columns]
    opt_used = [t for t in cleaned_weights if cleaned_weights[t] > 1e-6 and t in price_data.columns]

    # Normalize weights to sum to 1 for each portfolio
    user_w = np.array([portfolio_weights[t] for t in used])
    user_w = user_w / user_w.sum()
    opt_w = np.array([cleaned_weights[t] for t in opt_used])
    opt_w = opt_w / opt_w.sum()

    # Compute daily portfolio values (rebalance at start, no trading after)
    user_portfolio = (price_data[used] * user_w).sum(axis=1)
    opt_portfolio = (price_data[opt_used] * opt_w).sum(axis=1)

    bt_df = pd.DataFrame({
        "Original Portfolio": user_portfolio,
        "Optimized Portfolio": opt_portfolio
    })

    st.header("5. Backtest: Historical Portfolio Value")
    st.line_chart(bt_df)

    # --- Show backtest stats ---
    def backtest_stats(series):
        returns = series.pct_change().dropna()
        cumulative_return = (series.iloc[-1] / series.iloc[0]) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else np.nan
        running_max = np.maximum.accumulate(series)
        drawdown = (series - running_max) / running_max
        max_drawdown = drawdown.min()
        return {
            "Cumulative Return": f"{cumulative_return*100:.2f}%",
            "Annualized Volatility": f"{volatility*100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_drawdown*100:.2f}%"
        }

    user_stats = backtest_stats(user_portfolio)
    opt_stats = backtest_stats(opt_portfolio)

    stats_df = pd.DataFrame({
        "Metric": list(user_stats.keys()),
        "Original Portfolio": list(user_stats.values()),
        "Optimized Portfolio": list(opt_stats.values())
    })

    st.subheader("Backtest Performance Metrics")
    st.dataframe(stats_df)
