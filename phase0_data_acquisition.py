import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

SYMBOL = "ZOMATO.NS"
OUTPUT_DIR = r"C:\Stocks\ZOMATO LTD"

end_date = datetime.now()
start_date = end_date - relativedelta(years=5)

print(f"Downloading {SYMBOL}...")
print(f"Period: {start_date.date()} → {end_date.date()}")

stock_data = yf.download(
    SYMBOL,
    start=start_date,
    end=end_date,
    auto_adjust=False,
    progress=False
)

if stock_data.empty:
    raise RuntimeError(
        "Failed to retrieve data. "
        "Check the ticker symbol or network connection."
    )

# Handle yfinance MultiIndex columns
if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = stock_data.columns.get_level_values(0)

stock_data = stock_data.reset_index()
stock_data = stock_data.sort_values("Date").reset_index(drop=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_filename = os.path.join(
    OUTPUT_DIR,
    f"{SYMBOL}_5year_history.csv"
)

stock_data.to_csv(csv_filename, index=False)

print(f"Downloaded {len(stock_data)} observations.")
print(f"Saved to: {csv_filename}")
