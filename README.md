# Zomato Stock Analysis & LSTM Forecasting

An end-to-end time-series analysis and forecasting project using historical
Zomato Limited (NSE: ZOMATO) stock data.

The project is divided into three phases:

1. **Phase 0 — Data Acquisition**
2. **Phase 1 — Exploratory Data Analysis**
3. **Phase 2 — LSTM Forecasting & Model Evaluation**

The project does not evaluate the LSTM solely based on how attractive its
future forecast looks. Instead, it compares the model against a naive
baseline using out-of-sample walk-forward backtesting and incorporates
uncertainty into the final forecast.

---

## Project Pipeline

```text
                         ZOMATO.NS
                             │
                             ▼
              ┌──────────────────────────┐
              │         PHASE 0          │
              │     Data Acquisition     │
              │                          │
              │       Yahoo Finance      │
              │         yfinance         │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │         PHASE 1          │
              │  Exploratory Analysis    │
              │                          │
              │ • Price trends           │
              │ • Daily returns          │
              │ • Volatility             │
              │ • Moving averages        │
              │ • Yearly performance     │
              │ • Monthly returns        │
              │ • Calendar patterns      │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │         PHASE 2          │
              │    LSTM Forecasting      │
              │                          │
              │ • Walk-forward testing   │
              │ • Naive baseline         │
              │ • LSTM evaluation        │
              │ • Recursive forecasting  │
              │ • Monte-Carlo dropout    │
              │ • Reliability check      │
              │ • Signal generation      │
              └──────────────────────────┘
```


## Phase 0 — Data Acquisition

Historical Zomato stock data is retrieved from Yahoo Finance using
[`yfinance`](https://pypi.org/project/yfinance/).

The acquisition script:

- Downloads approximately five years of historical data
- Uses the NSE ticker `ZOMATO.NS`
- Validates whether data was successfully retrieved
- Saves the resulting dataset as a CSV file

### Input

```text
ZOMATO.NS
```
### Output
```text
ZOMATO.NS_5year_history.csv
```

The generated CSV is then used by Phase 1 and Phase 2.

---

## Phase 1 — Exploratory Data Analysis

Phase 1 examines the historical behavior of the stock before applying
machine learning.

### Analyses Performed

#### 1. Historical Closing Price

The closing price is plotted over the available historical period to
visualize long-term price movement.

#### 2. Daily Returns

Daily percentage returns are calculated as:

```text
Daily Return = (Today's Close - Previous Close) / Previous Close
```

Daily returns provide a view of short-term price variability.

#### 3. Historical Volatility

The standard deviation of daily returns is calculated as a simple measure
of historical volatility.

#### 4. Moving Averages

Two moving averages are calculated:

- 50-day moving average
- 200-day moving average

These are used to visualize medium- and long-term price trends.

#### 5. Yearly Performance

The first and last available closing prices of each calendar year are used
to calculate yearly price returns.

#### 6. Monthly Returns

Monthly compounded returns are calculated from daily returns.

#### 7. Monthly Return Heatmap

Monthly returns are displayed using a year-by-month heatmap to identify
periods of relatively strong and weak performance.

#### 8. Calendar-Based Analysis

Average closing prices are grouped by calendar month to explore potential
seasonal patterns.

> These analyses are exploratory and do not establish that observed
> patterns are statistically significant or predictive.

---

