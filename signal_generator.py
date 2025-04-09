#%%

import pandas as pd
import yfinance as yf
import ta
from datetime import datetime, timedelta

# --- 1. INPUTS --- #

# Your portfolio: tickers you own with cost basis and quantity
portfolio = {
    "F": {"cost_basis": 11.37, "quantity": 1.478943},
    "JEMA": {"cost_basis": 41.50, "quantity": 3.079218},
    "HLN": {"cost_basis": 9.99, "quantity": 2},
    "STLA": {"cost_basis":16.70, "quantity":2},
    "ORCL": {"cost_basis":136.99,"quantity":1},
    "KSPI": {"cost_basis":111.54,"quantity":3},
    "PH": {"cost_basis":633.36,"quantity":0.14},
    "BAYRY":{"cost_basis":6.48, "quantity": 2},
}

# Your universe of stocks to monitor (tickers only)
watchlist = ["MU", "FRHC", "CEE", "MAGS", "RUN", "AVGO", "NU", "GAP", "SHW", "SONY","ARM", "TCEHY","NVDA","AAPL","AMZN","AMD","AMC", "F", "JEMA","HLN", "STLA", "ORCL", "KSPI", "PH", "BAYRY"]

# --- 2. FUNCTIONS --- #

def get_rsi(ticker, period_days=14):
    end = datetime.now()
    start = end - timedelta(days=60)
    data = yf.download(ticker, start=start, end=end, progress=False)

    if data.empty or 'Close' not in data:
        return None, None
    try:
        rsi_ind = ta.momentum.RSIIndicator(data['Close'].squeeze(), window=period_days)
        data['RSI'] = rsi_ind.rsi()
    except ZeroDivisionError:
        return data['Close'].iloc[-1], None
    
    current_price = data['Close'].iloc[-1]
    current_rsi = data['RSI'].dropna().iloc[-1] if not data['RSI'].dropna().empty else None

    return current_price, current_rsi

# --- 3. ANALYSIS --- #

def analyze_portfolio():
    print("\n--- Sell Recommendations ---")
    for ticker, info in portfolio.items():
        price, rsi = get_rsi(ticker)
        if price is None:
            continue
        
        price = price.iloc[-1]
        

        cost = info['cost_basis']
        change_pct = (price - cost) / cost * 100

        if change_pct >= 5:
            print(f"Sell {ticker}: Up {change_pct:.2f}% from your cost basis.")
        elif rsi is not None and rsi > 70:
            print(f"Sell {ticker}: RSI is {rsi:.2f} (overbought).")

def analyze_watchlist():
    print("\n--- Buy Recommendations ---")
    for ticker in watchlist:
        price, rsi = get_rsi(ticker)
        if price is None or rsi is None:
            continue

        if rsi < 30:
            
            print(f"Buy {ticker}: RSI is {rsi:.2f} (oversold).")

# --- 4. RUN --- #
analyze_portfolio()
analyze_watchlist()

# %%
