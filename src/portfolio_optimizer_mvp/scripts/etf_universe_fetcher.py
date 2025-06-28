import yfinance as yf
import pandas as pd
from tqdm import tqdm

# Load ETF list (use your existing 200+ universe)
etfs = [
    "SPY", "QQQ", "VTI", "TLT", "GLD", "IWM", "BND", "IVV", "EFA", "VNQ",
    "XLF", "XLK", "XLY", "XLV", "XLI", "XLE", "XLB", "XLU", "XLC", "XBI",
    "ARKK", "DIA", "VEA", "VWO", "EEM", "EWW", "EWJ", "EWG", "FXI", "GDX",
    "GDXJ", "SLV", "USO", "UNG", "LQD", "HYG", "SHY", "IEF", "TLH", "TIP",
    "AGG", "MUB", "BIL", "BSV", "BNDX", "EMB", "VCIT", "VCSH", "VGLT", "VGIT",
    "VGSH", "VOO", "VUG", "VTV", "VO", "VB", "VBR", "VTWO", "VTIAX", "VXUS",
    "SCHB", "SCHX", "SCHA", "SCHD", "SCHF", "SCHE", "SCHP", "SCHO", "SPSM",
    "SPLG", "SPLV", "SPYG", "SPYV", "RSP", "IVOO", "IJR", "IJS", "IJH", "IWR",
    "IWF", "IWD", "IWB", "ITOT", "MTUM", "QUAL", "USMV", "VYM", "HDV", "DVY",
    "SDY", "NOBL", "PFF", "SCHH", "VNQI", "REET", "REM", "FREL", "IYR", "RWR",
    "XLRE", "PSR", "ICF", "RWX", "FTEC", "IYW", "IXN", "SOXX", "SMH", "XSD",
    "FBCG", "ARKW", "ARKF", "ARKG", "ARKQ", "PRNT", "ROBO", "BOTZ", "HACK",
    "CIBR", "SKYY", "FIVG", "IBB", "XHE", "XHS", "XPH", "XAR", "ITA", "XME",
    "XOP", "OIH", "KBE", "KRE", "IAT", "IAI", "IGV", "XSW", "FDN", "PNQI",
    "BJK", "BETZ", "MJ", "YOLO", "TOKE", "CNBS", "UFO", "ESPO", "NERD", "ECLN",
    "ICLN", "PBW", "TAN", "FAN", "QCLN", "CRAK", "FRAK", "XLFV", "FREL", "FNCL"
]

# Define manual categories by keyword
CATEGORY_MAP = {
    "Bond": ["TLT", "BND", "SHY", "IEF", "LQD", "AGG", "TIP", "HYG"],
    "Equity - Broad": ["SPY", "QQQ", "VTI", "IVV", "IWB", "VOO", "ITOT"],
    "Commodity": ["GLD", "SLV", "USO", "UNG"],
    "REIT": ["VNQ", "SCHH", "IYR", "XLRE"],
    "Thematic": ["ARKK", "ARKG", "ARKW", "BOTZ", "ROBO", "HACK", "FIVG"],
    "Clean Energy": ["TAN", "ICLN", "PBW", "QCLN", "FAN"]
}

def assign_category(ticker):
    for category, symbols in CATEGORY_MAP.items():
        if ticker in symbols:
            return category
    return "Other"

# Fetch ETF descriptions
data = []
for symbol in tqdm(etfs):
    try:
        info = yf.Ticker(symbol).info
        name = info.get("longName") or ""
        category = assign_category(symbol)
        data.append({"Symbol": symbol, "Name": name, "Category": category})
    except Exception:
        data.append({"Symbol": symbol, "Name": "", "Category": "Other"})

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("../csv/etf_metadata.csv", index=False)
print("Saved etf_metadata.csv with category assignments.")
