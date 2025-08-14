import pandas as pd
import yfinance as yf
import ta
from datetime import datetime, timedelta
import gradio as gr
import io 
import contextlib
import sqlite3
from collections import defaultdict, deque


# --- 1. INPUTS --- #

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

watchlist = ["BRK-B","MU", "FRHC", "CEE", "MAGS", "RUN", "AVGO", "NU", "GAP", "SHW", "SONY","ARM", "TCEHY","NVDA","AAPL","AMZN","AMD","AMC", "F", "JEMA","HLN", "STLA", "ORCL", "KSPI", "PH", "BAYRY"]

# --- 2. FUNCTIONS --- #

def init_db():
    conn = sqlite3.connect("trades.db")
    cursor = conn.cursor()

    # Create trades table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            type TEXT NOT NULL, -- "buy" or "sell"
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create portfolio table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL UNIQUE,
            quantity REAL NOT NULL,
            cost_basis REAL NOT NULL,
            average_price REAL NOT NULL,  -- This can help track cost basis changes
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    #for ticker, data in portfolio.items():
    #    cost_basis = data["cost_basis"]
    #    quantity = data["quantity"]
    #    average_price = cost_basis  # Assume cost_basis is the average price initially
        
        # Insert or update the portfolio table
    #    cursor.execute("""
    #        INSERT OR REPLACE INTO portfolio (ticker, quantity, cost_basis, average_price)
    #        VALUES (?, ?, ?, ?)
    #   """, (ticker, quantity, cost_basis, average_price))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()


def df_or_message(df, message="No recommendations for now😴"):
    return df if not df.empty else message

def backfill_portfolio(portfolio):
    conn = sqlite3.connect("trades.db")
    c = conn.cursor()
    default_date = "2025-01-01 00:00:00"

    for ticker, data in portfolio.items():
        c.execute('''
            INSERT INTO trades (ticker, quantity, price, type, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (ticker, data['quantity'], data['cost_basis'], "BUY", default_date))
    
    conn.commit()
    conn.close()

# Run this ONCE to backfill
#backfill_portfolio(portfolio)

    
def log_trade(trade_type, ticker, quantity, price):
    conn = sqlite3.connect("trades.db")
    c = conn.cursor()
    
    # Get the current time and format it to remove the 'T' from ISO format
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # '2025-04-15 02:41:51'

    # Insert the trade into the database with the formatted timestamp
    c.execute('''
        INSERT INTO trades (ticker, quantity, price, type, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (ticker, quantity, price, trade_type, timestamp))
    
    conn.commit()
    print(view_all_trades())
    conn.close()

def view_all_trades():
    conn = sqlite3.connect("trades.db")
    df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp", conn)
    conn.close()
    return df

def remove_exact_duplicates():
    conn = sqlite3.connect("trades.db")
    c = conn.cursor()
    c.execute("""
        DELETE FROM trades
        WHERE rowid NOT IN (
            SELECT MIN(rowid)
            FROM trades
            GROUP BY ticker, quantity, price, type, timestamp
        );
    """)
    conn.commit()
    conn.close()

#remove_exact_duplicates()
def remove_all_sell_trades():
    # Connect to the database
    conn = sqlite3.connect("trades.db")
    c = conn.cursor()

    # Delete all sell trades from the trades table
    c.execute("DELETE FROM trades WHERE type = 'SELL'")

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    print("All sell trades have been removed.")

# Example usage
#remove_all_sell_trades()
print(view_all_trades())

def calculate_realized_profits(start_date=None, end_date=None):
    # If no dates are provided, use the whole period
    if not start_date:
        start_date = datetime.min  # Earliest date possible
    if not end_date:
        end_date = datetime.now()  # Current time

    # Convert to string format for the query
    start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_date_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
    #print(end_date_str)

    conn = sqlite3.connect("trades.db")
    c = conn.cursor()
    
    # Query trades within the specified date range
    c.execute('SELECT * FROM trades WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp', (start_date_str, end_date_str))
    #c.execute('SELECT * FROM trades')
    trades = c.fetchall()

    # Query the current portfolio from the database
    c.execute('SELECT ticker, quantity, cost_basis FROM portfolio')
    portfolio = {ticker: {"quantity": quantity, "cost_basis": cost_basis} for ticker, quantity, cost_basis in c.fetchall()}

    conn.close()

    realized = defaultdict(float)  # {ticker: realized profit}
    lots = defaultdict(deque)  # {ticker: deque of (qty, price)}

    print(f"Queried {len(trades)} trades from {start_date_str} to {end_date_str}")

    for _, ticker, quantity, price, trade_type, timestamp in trades:
        quantity = float(quantity)
        price = float(price)

        if trade_type == 'BUY':
            lots[ticker].append((quantity, price))

        elif trade_type == 'SELL':
            to_sell = quantity
            while to_sell > 0 and lots[ticker]:
                lot_qty, lot_price = lots[ticker][0]
                matched_qty = min(to_sell, lot_qty)
                profit = (price - lot_price) * matched_qty
                realized[ticker] += profit

                # Reduce or remove lot
                if matched_qty == lot_qty:
                    lots[ticker].popleft()
                else:
                    lots[ticker][0] = (lot_qty - matched_qty, lot_price)

                to_sell -= matched_qty

    # Convert to DataFrame
    return pd.DataFrame([
        {"Ticker": ticker, "Realized Profit": round(profit, 2)}
        for ticker, profit in realized.items()
    ])




def buy_stock(ticker, quantity, price):
    conn = sqlite3.connect("trades.db")
    cursor = conn.cursor()

    # Calculate the cost of the purchase
    quantity = float(quantity)
    cost = price * quantity

    # Check if the stock exists in the portfolio
    cursor.execute("SELECT quantity, cost_basis FROM portfolio WHERE ticker = ?", (ticker,))
    result = cursor.fetchone()

    if result:
        # If stock exists, update the quantity and cost basis
        old_qty, old_cost = result
        new_qty = old_qty + quantity
        new_cost_basis = ((old_cost * old_qty) + cost) / new_qty

        cursor.execute("""
            UPDATE portfolio
            SET quantity = ?, cost_basis = ?, last_updated = CURRENT_TIMESTAMP
            WHERE ticker = ?
        """, (new_qty, new_cost_basis, ticker))

    else:
        # If stock does not exist, insert a new record
        log_trade("BUY", ticker, quantity, price)

    # Log the purchase in the trades table
    cursor.execute("""
        INSERT INTO trades (ticker, quantity, price, type, timestamp)
        VALUES (?, ?, ?, 'BUY', CURRENT_TIMESTAMP)
    """, (ticker, quantity, price))

    conn.commit()
    conn.close()

    return f"Bought {quantity:.4f} shares of {ticker} at ${price:.2f}"



def sell_stock(ticker, quantity, price):
    conn = sqlite3.connect("trades.db")
    cursor = conn.cursor()

    # Check if the stock exists in the portfolio
    cursor.execute("SELECT quantity FROM portfolio WHERE ticker = ?", (ticker,))
    result = cursor.fetchone()

    if not result:
        conn.close()
        return f"Cannot sell {ticker}: Not in portfolio."

    owned_qty = result[0]
    print(f"Owned quantity of {ticker}: {owned_qty}")
    if quantity > owned_qty:
        conn.close()
        return f"Cannot sell {quantity:.4f} shares — only {owned_qty:.4f} owned."

    # Calculate the proceeds
    proceeds = price * quantity

    # Log the sell transaction into the trades table
    log_trade("SELL", ticker, quantity, price)

    # Update the portfolio: reduce the quantity of the stock
    new_quantity = owned_qty - quantity
    if new_quantity > 0:
        cursor.execute("""
            UPDATE portfolio
            SET quantity = ?, last_updated = CURRENT_TIMESTAMP
            WHERE ticker = ?
        """, (new_quantity, ticker))
    else:
        # If no shares are left, delete the stock from the portfolio
        cursor.execute("DELETE FROM portfolio WHERE ticker = ?", (ticker,))

    conn.commit()
    conn.close()

    return f"Sold {quantity:.4f} shares of {ticker} at ${price:.2f} — proceeds ${proceeds:.2f}"


#buy sell signals 
def fetch_market_data(tickers, lookback_days=80):
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    data = yf.download(tickers, start=start, end=end, group_by="ticker", progress=False)
    return data
tickers = list(set(portfolio.keys()).union(set(watchlist)))

#get the data for all tickers for max horizon we need 
data = fetch_market_data(tickers) 

def get_rsi(data, ticker, period_days=14):
    if ticker not in data or data[ticker].empty or 'Close' not in data[ticker]:
        return None, None
    series = data[ticker]['Close'].dropna()
    if series.empty:
        return None, None
    try:
        rsi_ind = ta.momentum.RSIIndicator(series, window=period_days)
        rsi_values = rsi_ind.rsi()
    except ZeroDivisionError:
        return series.iloc[-1], None
    if rsi_values.dropna().empty:
        return None, None
    return series.iloc[-1], rsi_values.dropna().iloc[-1]

def get_ma_signals(data, ticker, short_window=20, long_window=50):
    if ticker not in data or data[ticker].empty or 'Close' not in data[ticker]:
        return None
    series = data[ticker]['Close'].dropna()
    if len(series) < long_window:
        return None
    short_ma = series.rolling(window=short_window).mean()
    long_ma = series.rolling(window=long_window).mean()
    short_prev = short_ma.iloc[-2]
    long_prev = long_ma.iloc[-2]
    short_now = short_ma.iloc[-1]
    long_now = long_ma.iloc[-1]
    if short_prev < long_prev and short_now > long_now:
        return "Buy"
    elif short_prev > long_prev and short_now < long_now:
        return "Sell"
    else:
        return None
    
def get_price_history(data, ticker, days=3):
    if ticker not in data or data[ticker].empty or 'Close' not in data[ticker]:
        return None
    return data[ticker]['Close'].dropna().tail(days)

# --- 3. SIGNAL FUNCTIONS --- #

def analyze_sell_ma(data):
    rows = []
    for ticker, info in portfolio.items():
        ma_signal = get_ma_signals(data,ticker)
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


def analyze_sell_rsi(data):
    rows = []
    for ticker, info in portfolio.items():
        price, rsi = get_rsi(data, ticker)
        if price is None or rsi is None:
            continue
        if rsi > 70:
            cost = portfolio[ticker]['cost_basis']
            quantity = portfolio[ticker]['quantity']
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


def analyze_sell_pct(data):
    rows = []
    for ticker, info in portfolio.items():
        price, _ = get_rsi(data,ticker)
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


def analyze_buy_ma(data):
    rows = []
    for ticker in data.columns.levels[0]:
        signal = get_ma_signals(data, ticker)
        if signal == "Buy":
            rows.append({"Ticker": ticker})
    return pd.DataFrame(rows)

def analyze_buy_rsi(data):
    rows = []
    for ticker in data.columns.levels[0]:
        price, rsi = get_rsi(data,ticker)
        if price is None or rsi is None:
            continue
        if rsi < 50:
            rows.append({"Ticker": ticker, "RSI": round(rsi, 2)})
    return pd.DataFrame(rows)

def analyze_buy_pct(data):
    rows = []
    for ticker in data.columns.levels[0]:
        try:
            prices = get_price_history(data, ticker, days=3)
            if prices is not None and len(prices) >= 2:
                old_price = prices.iloc[0]
                recent_price = prices.iloc[-1]
                change_pct = (recent_price - old_price) / old_price * 100

                if change_pct < 0:
                    rows.append({
                        "Ticker": ticker,
                        "% Change (3d)": round(change_pct, 2)
                    })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    return pd.DataFrame(rows)


def get_portfolio_df():
    return pd.DataFrame.from_dict(portfolio, orient='index').reset_index().rename(columns={"index": "Ticker"})

def get_portfolio():
    conn = sqlite3.connect("trades.db")
    c = conn.cursor()
    c.execute('SELECT ticker, quantity, cost_basis FROM portfolio')
    portfolio = {ticker: {"quantity": quantity, "cost_basis": cost_basis} for ticker, quantity, cost_basis in c.fetchall()}
    portfolio_df = pd.DataFrame.from_dict(portfolio, orient='index').reset_index().rename(columns={"index": "Ticker"})

    conn.close()
    return portfolio_df

# --- 4. GRADIO INTERFACE --- #

def run_analysis():
    sell_ma_output = analyze_sell_ma(data)
    sell_rsi_output = analyze_sell_rsi(data)
    sell_pct_output = analyze_sell_pct(data)
    buy_ma_output = analyze_buy_ma(data)
    buy_rsi_output = analyze_buy_rsi(data)
    buy_pct_output = analyze_buy_pct(data)

    return (
        df_or_message(sell_ma_output),
        df_or_message(sell_rsi_output),
        df_or_message(sell_pct_output),
        df_or_message(buy_ma_output),
        df_or_message(buy_rsi_output),
        df_or_message(buy_pct_output)
    )
    
#init_db()

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
    portfolio_table = gr.Dataframe(value=get_portfolio(), interactive=False)

    def refresh_portfolio():
        return get_portfolio()

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

    with gr.Row():
        # Option 1: Use a textbox with HTML date input
        start_date_picker = gr.Textbox(
            label="Start Date", 
            value=datetime.now().strftime('%Y-%m-%d'),
            elem_id="start_date"
        )
        
        end_date_picker = gr.Textbox(
            label="End Date",
            value=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  
            elem_id="end_date"
        )

    with gr.Row():
        profit_btn = gr.Button("Show Realized Profits")
        profit_table = gr.Dataframe(label="Realized Profits")

    # --- 3. Update Button Click Function --- #

    def show_realized_profits(start_date, end_date):
        # Convert the selected date values into datetime objects
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
        
        # Call the function with the selected date range
        profits = calculate_realized_profits(start_date=start_date_obj, end_date=end_date_obj)
        print(profits)
        return profits

    # Connect the button click to the function
    profit_btn.click(fn=show_realized_profits, 
                    inputs=[start_date_picker, end_date_picker], 
                    outputs=profit_table)



demo.launch()
