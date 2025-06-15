import yfinance as yf
import pandas as pd
from tqdm import tqdm

# Get list of current S&P 500 tickers from Wikipedia
wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
tables = pd.read_html(wiki_url)
df_raw = tables[0]

# Extract tickers and metadata
df = df_raw[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
df.columns = ["Ticker", "Company", "Sector", "Industry"]

# Clean tickers (some have dots instead of dashes)
df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)

# Fetch long names via yfinance
data = []
for ticker in tqdm(df["Ticker"].tolist()):
    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or ""
    except Exception:
        name = ""
    data.append(name)

df["Yahoo Name"] = data

# Save as CSV
df.to_csv("../csv/sp500_secmaster.csv", index=False)
print("Saved sp500_secmaster.csv with metadata.")
