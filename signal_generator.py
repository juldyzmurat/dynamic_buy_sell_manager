import pandas as pd
import yfinance as yf
import ta
from datetime import datetime, timedelta
import gradio as gr
import io 
import contextlib

# --- 1. INPUTS --- #

portfolio = {
    "F": {"cost_basis": 11.37, "quantity": 1.478943},
    "JEMA": {"cost_basis": 41.50, "quantity": 3.079218},
    "HLN": {"cost_basis": 9.99, "quantity": 2},
    "STLA": {"cost_basis":16.70, "quantity":2},
    "ORCL": {"cost_basis":136.99,"quantity":1},
    "KSPI": {"cost_basis":60.54,"quantity":3},
    "PH": {"cost_basis":633.36,"quantity":0.14},
    "BAYRY":{"cost_basis":6.48, "quantity": 2},
}

watchlist = ["MU", "FRHC", "CEE", "MAGS", "RUN", "AVGO", "NU", "GAP", "SHW", "SONY","ARM", "TCEHY","NVDA","AAPL","AMZN","AMD","AMC", "F", "JEMA","HLN", "STLA", "ORCL", "KSPI", "PH", "BAYRY"]

# --- 2. FUNCTIONS --- #

def buy_stock(ticker, quantity, price):
    
    quantity = float(quantity)
    cost = price * quantity
    if ticker in portfolio:
        old_qty = portfolio[ticker]["quantity"]
        old_cost = portfolio[ticker]["cost_basis"]
        new_qty = old_qty + quantity
        new_cost_basis = ((old_cost * old_qty) + cost) / new_qty
        portfolio[ticker] = {"cost_basis": new_cost_basis, "quantity": new_qty}
    else:
        portfolio[ticker] = {"cost_basis": price, "quantity": quantity}
    return f"Bought {quantity:.4f} shares of {ticker} at ${price:.2f}"


def sell_stock(ticker, quantity, price):
    if ticker not in portfolio:
        return f"Cannot sell {ticker}: Not in portfolio."
    quantity = portfolio[ticker]["quantity"]
    proceeds = price * quantity
    del portfolio[ticker]
    return f"Sold all {quantity:.4f} shares of {ticker} at ${price:.2f} — proceeds ${proceeds:.2f}"


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

def get_ma_signals(ticker, short_window=20, long_window=50):
    end = datetime.now()
    start = end - timedelta(days=long_window+30)
    data = yf.download(ticker, start=start, end=end, progress=False)
    
    if data.empty or 'Close' not in data:
        return None
    
    short_ma = data['Close'].rolling(window=short_window).mean()
    long_ma = data['Close'].rolling(window=long_window).mean()
    
    short_prev = short_ma.iloc[-2].item()
    long_prev = long_ma.iloc[-2].item()
    short_now = short_ma.iloc[-1].item()
    long_now = long_ma.iloc[-1].item()
    
    if short_prev < long_prev and short_now > long_now:
        return "Buy"
    elif short_prev > long_prev and short_now < long_now:
        return "Sell"
    else:
        return None

# --- 3. SIGNAL FUNCTIONS --- #

def analyze_sell_ma():
    rows = []
    for ticker, info in portfolio.items():
        ma_signal = get_ma_signals(ticker)
        if ma_signal == "Sell":
            price, _ = get_rsi(ticker)
            if price is None:
                continue
            cost = info['cost_basis']
            quantity = info['quantity']
            change_pct = (price - cost) / cost * 100
            gain = (price - cost) * quantity
            preview = f"${gain:.2f} at ${price:.2f}"
            rows.append({
                "Ticker": ticker,
                "Signal": "MA Crossover",
                "% Change": round(change_pct, 2),
                "Preview Sale": preview
            })
    return pd.DataFrame(rows)


def analyze_sell_rsi():
    rows = []
    for ticker, info in portfolio.items():
        price, rsi = get_rsi(ticker)
        if price is None or rsi is None:
            continue
        if rsi > 70:
            cost = info['cost_basis']
            quantity = info['quantity']
            change_pct = (price - cost) / cost * 100
            gain = (price - cost) * quantity
            preview = f"${gain:.2f} at ${price:.2f}"
            rows.append({
                "Ticker": ticker,
                "RSI": round(rsi, 2),
                "% Change": round(change_pct, 2),
                "Preview Sale": preview
            })
    return pd.DataFrame(rows)


def analyze_sell_pct():
    rows = []
    for ticker, info in portfolio.items():
        price, _ = get_rsi(ticker)
        if price is not None:
            price = price.item()  # Ensures the value is a Python float
            cost = info['cost_basis']
            quantity = info['quantity']
            change_pct = (price - cost) / cost * 100
            if change_pct >= 5:
                gain = (price - cost) * quantity
                preview = f"${gain:.2f} gain at ${price:.2f}"
                rows.append({
                    "Ticker": ticker,
                    "% Change": round(change_pct, 2),
                    "Preview Sale": preview
                })
    return pd.DataFrame(rows)


def analyze_buy_ma():
    rows = []
    for ticker in watchlist:
        ma_signal = get_ma_signals(ticker)
        if ma_signal == "Buy":
            rows.append({"Ticker": ticker})
    return pd.DataFrame(rows)

def analyze_buy_rsi():
    buy_rsi_signals = []
    for ticker in watchlist:
        price, rsi = get_rsi(ticker)
        if price is None or rsi is None:
            continue
        if rsi < 50:
            buy_rsi_signals.append({"Ticker": ticker, "RSI": rsi})
    
    return pd.DataFrame(buy_rsi_signals) if buy_rsi_signals else pd.DataFrame(columns=["Ticker", "RSI"])

def analyze_buy_pct():
    rows = []
    for ticker in watchlist:
        price, _ = get_rsi(ticker)
        if price.item() and ticker in portfolio:
            cost = portfolio[ticker]['cost_basis']
            change_pct = (price.item() - cost) / cost * 100
            if change_pct < -5:
                rows.append({"Ticker": ticker, "% Change": round(change_pct, 2)})
    return pd.DataFrame(rows)

def get_portfolio_df():
    return pd.DataFrame.from_dict(portfolio, orient='index').reset_index().rename(columns={"index": "Ticker"})


# --- 4. GRADIO INTERFACE --- #

def run_analysis():
    sell_ma_output = analyze_sell_ma()
    sell_rsi_output = analyze_sell_rsi()
    sell_pct_output = analyze_sell_pct()
    buy_ma_output = analyze_buy_ma()
    buy_rsi_output = analyze_buy_rsi()
    buy_pct_output = analyze_buy_pct()

    return sell_ma_output, sell_rsi_output, sell_pct_output, buy_ma_output, buy_rsi_output, buy_pct_output

with gr.Blocks() as demo:
    gr.Markdown("## 📈 Portfolio Signal Generator")

    with gr.Row():
        run_btn = gr.Button("Run Analysis")

    with gr.Row():
        sell_ma_box = gr.Textbox(label="Sell MA")
        sell_rsi_box = gr.Textbox(label="Sell RSI")
        sell_pct_box = gr.Dataframe(label="Sell PCT")
    
    with gr.Row():
        buy_ma_box = gr.Textbox(label="Buy MA")
        buy_rsi_box = gr.Dataframe(label="Buy RSI")
        buy_pct_box = gr.Dataframe(label="Buy PCT")

    run_btn.click(fn=run_analysis,
                outputs=[sell_ma_box, sell_rsi_box, sell_pct_box,
                        buy_ma_box, buy_rsi_box, buy_pct_box])

    gr.Markdown("## 💼 Current Portfolio")
    
    refresh_btn = gr.Button("Refresh Portfolio")
    portfolio_table = gr.Dataframe(value=get_portfolio_df(), interactive=False)

    def refresh_portfolio():
        return get_portfolio_df()

    refresh_btn.click(fn=refresh_portfolio, outputs=portfolio_table)

    gr.Markdown("## 🔁 Manual Trade")

    with gr.Accordion("Buy / Sell Stocks", open=False):
        with gr.Row():
            trade_ticker = gr.Textbox(label="Ticker")
            trade_quantity = gr.Number(label="Quantity", precision=4)
            trade_price  = gr.Number(label="Price", precision=2)

        with gr.Row():
            buy_btn = gr.Button("Buy")
            sell_btn = gr.Button("Sell")

        with gr.Row():
            buy_result = gr.Textbox(label="Buy Result", interactive=False)
            sell_result = gr.Textbox(label="Sell Result", interactive=False)

        buy_btn.click(fn=buy_stock, inputs=[trade_ticker, trade_quantity, trade_price], outputs=buy_result)
        sell_btn.click(fn=sell_stock, inputs=[trade_ticker,trade_quantity, trade_price], outputs=sell_result)



demo.launch()

# %%
