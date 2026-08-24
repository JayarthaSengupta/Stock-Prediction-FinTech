"""
Zomato Stock Analysis — Phase 1
================================

Exploratory and statistical analysis of Zomato (NSE: ZOMATO)
historical stock data.

Phase 1 covers:
    1. Dataset inspection and cleaning
    2. Descriptive statistics
    3. Historical closing-price trend
    4. Daily returns and volatility
    5. 50-day and 200-day moving averages
    6. Yearly performance
    7. Monthly return analysis
    8. Monthly return heatmap
    9. Calendar-based seasonal analysis

This phase is exploratory. The forecasting model is implemented
separately in Phase 2.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ================================================================
# CONFIGURATION
# ================================================================

FILE_PATH = r"C:\Stocks\ZOMATO LTD\ZOMATO.NS_5year_history.csv"


# ================================================================
# 1. LOAD & CLEAN DATA
# ================================================================

print("=" * 70)
print("ZOMATO STOCK ANALYSIS — PHASE 1")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(FILE_PATH)

# Check required columns
required_columns = ["Date", "Close"]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required column(s): {missing_columns}"
    )

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Convert Close to numeric
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

# Remove invalid rows
df = df.dropna(subset=["Date", "Close"])

# Sort chronologically
# This is essential for all time-series calculations.
df = df.sort_values("Date").reset_index(drop=True)

print(f"\nLoaded {len(df)} observations.")
print(
    f"Date range: "
    f"{df['Date'].min().date()} → {df['Date'].max().date()}"
)

print("\nDataset columns:")
print(df.columns.tolist())


# ================================================================
# 2. DATASET INSPECTION
# ================================================================

print("\n" + "=" * 70)
print("DATASET PREVIEW")
print("=" * 70)

print("\nFirst 5 observations:")
print(df.head())

print("\nLast 5 observations:")
print(df.tail())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())


# ================================================================
# 3. DESCRIPTIVE STATISTICS
# ================================================================

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

summary_stats = df.describe()

print(summary_stats)


# ================================================================
# 4. HISTORICAL CLOSE PRICE TREND
# ================================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["Date"],
    df["Close"],
    label="Close Price"
)

plt.title(
    "Zomato (NSE: ZOMATO) — Historical Closing Price"
)

plt.xlabel("Date")
plt.ylabel("Close Price (INR)")

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()


# ================================================================
# 5. DAILY RETURNS
# ================================================================

# Percentage change from one trading day to the next.
df["Daily Return"] = df["Close"].pct_change()

# Remove the first NaN created by pct_change()
valid_returns = df["Daily Return"].dropna()

# Historical volatility
volatility = valid_returns.std()

print("\n" + "=" * 70)
print("DAILY RETURN & VOLATILITY ANALYSIS")
print("=" * 70)

print(
    f"\nDaily return mean: "
    f"{valid_returns.mean() * 100:.4f}%"
)

print(
    f"Daily return standard deviation (volatility): "
    f"{volatility * 100:.4f}%"
)

print(
    f"Maximum daily return: "
    f"{valid_returns.max() * 100:.2f}%"
)

print(
    f"Minimum daily return: "
    f"{valid_returns.min() * 100:.2f}%"
)


# Plot daily returns

plt.figure(figsize=(12, 6))

plt.plot(
    df["Date"],
    df["Daily Return"],
    label="Daily Return"
)

plt.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)

plt.title(
    "Zomato — Daily Returns"
)

plt.xlabel("Date")
plt.ylabel("Daily Return")

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()


# ================================================================
# 6. MOVING AVERAGES
# ================================================================

# 50-day moving average
df["50_MA"] = df["Close"].rolling(window=50).mean()

# 200-day moving average
df["200_MA"] = df["Close"].rolling(window=200).mean()

print("\n" + "=" * 70)
print("MOVING AVERAGE ANALYSIS")
print("=" * 70)

print(
    f"\nLatest Close: "
    f"₹{df['Close'].iloc[-1]:.2f}"
)

print(
    f"Latest 50-Day MA: "
    f"₹{df['50_MA'].iloc[-1]:.2f}"
)

print(
    f"Latest 200-Day MA: "
    f"₹{df['200_MA'].iloc[-1]:.2f}"
)


# Plot moving averages

plt.figure(figsize=(12, 6))

plt.plot(
    df["Date"],
    df["Close"],
    label="Close Price"
)

plt.plot(
    df["Date"],
    df["50_MA"],
    label="50-Day Moving Average"
)

plt.plot(
    df["Date"],
    df["200_MA"],
    label="200-Day Moving Average"
)

plt.title(
    "Zomato — Closing Price with 50-Day and 200-Day Moving Averages"
)

plt.xlabel("Date")
plt.ylabel("Price (INR)")

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()


# ================================================================
# 7. YEARLY PERFORMANCE
# ================================================================

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# First and last trading-day close of each year
yearly_performance = (
    df.groupby("Year")["Close"]
    .agg(["first", "last"])
)

# Calculate yearly return
yearly_performance["Yearly Return (%)"] = (
    (
        yearly_performance["last"]
        - yearly_performance["first"]
    )
    / yearly_performance["first"]
) * 100

print("\n" + "=" * 70)
print("YEARLY PERFORMANCE")
print("=" * 70)

print(
    yearly_performance.round(2)
)


# ================================================================
# 8. MONTHLY RETURNS
# ================================================================

# Calculate monthly compounded returns from daily returns.
#
# For each month:
#
# Monthly Return = PRODUCT(1 + daily returns) - 1
#
# This is preferable to simply averaging daily returns.

monthly_returns = (
    df.set_index("Date")["Daily Return"]
    .groupby(pd.Grouper(freq="ME"))
    .apply(lambda x: (1 + x.dropna()).prod() - 1)
)

monthly_returns = monthly_returns.dropna()

monthly_returns_df = monthly_returns.to_frame(
    name="Monthly Return"
)

monthly_returns_df["Year"] = monthly_returns_df.index.year
monthly_returns_df["Month"] = monthly_returns_df.index.month

print("\n" + "=" * 70)
print("MONTHLY RETURN ANALYSIS")
print("=" * 70)

print(
    "\nMonthly return statistics:"
)

print(
    monthly_returns_df["Monthly Return"]
    .describe()
    .to_frame()
    .T
)


# ================================================================
# 9. MONTHLY RETURN HEATMAP
# ================================================================

# Convert monthly returns into:
#
# Rows    → Years
# Columns → Months
# Values  → Monthly percentage returns

monthly_heatmap = monthly_returns_df.pivot_table(
    index="Year",
    columns="Month",
    values="Monthly Return",
    aggfunc="first"
) * 100

# Rename month columns
month_names = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec"
}

monthly_heatmap = monthly_heatmap.rename(
    columns=month_names
)

plt.figure(figsize=(12, 6))

sns.heatmap(
    monthly_heatmap,
    annot=True,
    cmap="RdYlGn",
    center=0,
    fmt=".2f",
    linewidths=0.5
)

plt.title(
    "Zomato — Monthly Returns (%)"
)

plt.xlabel("Month")
plt.ylabel("Year")

plt.tight_layout()

plt.show()


# ================================================================
# 10. CALENDAR-BASED SEASONAL ANALYSIS
# ================================================================

# Instead of calculating average monthly PRICE,
# calculate average monthly RETURN.
#
# This is more meaningful when looking for calendar effects
# because stock prices themselves are non-stationary.

monthly_avg_return = (
    df.groupby("Month")["Daily Return"]
    .mean()
    * 100
)

monthly_avg_return = monthly_avg_return.rename(
    index=month_names
)

print("\n" + "=" * 70)
print("AVERAGE MONTHLY DAILY RETURN")
print("=" * 70)

print(
    monthly_avg_return.round(4)
)


# Plot average monthly return

plt.figure(figsize=(10, 6))

monthly_avg_return.plot(
    kind="bar"
)

plt.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)

plt.title(
    "Zomato — Average Daily Return by Calendar Month"
)

plt.xlabel("Month")
plt.ylabel("Average Daily Return (%)")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ================================================================
# 11. ADDITIONAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("PHASE 1 SUMMARY")
print("=" * 70)

total_return = (
    (df["Close"].iloc[-1] - df["Close"].iloc[0])
    / df["Close"].iloc[0]
) * 100

best_year = yearly_performance["Yearly Return (%)"].idxmax()
worst_year = yearly_performance["Yearly Return (%)"].idxmin()

best_year_return = yearly_performance.loc[
    best_year,
    "Yearly Return (%)"
]

worst_year_return = yearly_performance.loc[
    worst_year,
    "Yearly Return (%)"
]

print(
    f"\nOverall price change: "
    f"{total_return:+.2f}%"
)

print(
    f"Best calendar year: "
    f"{best_year} ({best_year_return:+.2f}%)"
)

print(
    f"Worst calendar year: "
    f"{worst_year} ({worst_year_return:+.2f}%)"
)

print(
    f"Daily return volatility: "
    f"{volatility * 100:.4f}%"
)

print(
    f"Latest closing price: "
    f"₹{df['Close'].iloc[-1]:.2f}"
)

print("\nPhase 1 analysis completed.")
print("=" * 70)
