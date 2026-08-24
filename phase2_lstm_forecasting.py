"""
LSTM Stock Return Forecaster
============================

A more rigorous LSTM-based stock forecasting pipeline.

What this script does:
    1. Loads historical closing prices.
    2. Converts prices into daily percentage returns.
    3. Trains an LSTM to predict the next day's return.
    4. Performs an expanding-window walk-forward backtest.
    5. Compares the LSTM against a naive "tomorrow = today" baseline.
    6. Produces a 30-trading-day recursive forecast.
    7. Runs Monte-Carlo dropout to estimate a distribution of outcomes.
    8. Calculates:
        - Expected 30-day return
        - 10th / 90th percentile outcomes
        - Probability of a positive outcome
        - Backtest MAE / RMSE / MAPE
        - Directional accuracy
        - Model vs naive baseline
    9. Produces a BUY / HOLD / AVOID signal only when the model
       demonstrates sufficient historical performance.

IMPORTANT:
    This is a forecasting experiment, NOT a financial adviser.

    Stock prices are highly noisy and affected by information that a
    price-only LSTM cannot know in advance.

    Monte-Carlo dropout provides a distribution of model outputs, not
    a statistically guaranteed prediction interval.

    A strong historical backtest does not guarantee future profitability.
"""

import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

# ===============================================================
# CONFIG
# ===============================================================

CSV_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else r"C:\Stocks\ITI\ITI.NS_2year_history.csv"
)

LOOKBACK = 60

FORECAST_HORIZON = 30

# Number of actual historical observations reserved for testing.
TEST_DAYS = 60

# Minimum expected return required before BUY can be considered.
BUY_THRESHOLD = 5.0

# Minimum probability that the 30-day outcome is positive.
MIN_PROBABILITY_POSITIVE = 65.0

# Monte-Carlo dropout simulations.
MC_PASSES = 200

# Initial model training.
EPOCHS = 100

# Walk-forward retraining is intentionally lighter.
WF_EPOCHS = 40

BATCH_SIZE = 32

# Retrain the model every N walk-forward observations.
# Lower = more adaptive but considerably slower.
RETRAIN_EVERY = 5

SEED = 42

OUTPUT_PLOT = "lstm_forecast.png"


# ===============================================================
# REPRODUCIBILITY
# ===============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)

warnings.filterwarnings("ignore")


# ===============================================================
# 1. LOAD & CLEAN DATA
# ===============================================================


def load_close_series(path):
    """
    Load Date + Close from a CSV file.

    Expected columns:
        Date
        Close
    """

    df = pd.read_csv(path)

    required_columns = {"Date", "Close"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = (
        df[["Date", "Close"]]
        .dropna()
        .sort_values("Date")
        .drop_duplicates(subset="Date")
        .reset_index(drop=True)
    )

    if len(df) == 0:
        raise ValueError("No valid Date/Close observations found.")

    if (df["Close"] <= 0).any():
        raise ValueError("Close prices must be positive.")

    return df


# ===============================================================
# 2. CREATE RETURNS
# ===============================================================


def add_returns(df):
    """
    Calculate daily percentage returns.

    Return[t] = Close[t] / Close[t-1] - 1
    """

    df = df.copy()

    df["Return"] = df["Close"].pct_change()

    df = df.dropna().reset_index(drop=True)

    if df["Return"].isna().any():
        raise ValueError("Return calculation produced NaN values.")

    return df


# ===============================================================
# 3. SEQUENCE BUILDING
# ===============================================================


def make_sequences(values, lookback):
    """
    Convert a 1D time series into LSTM sequences.

    Example:

        [r1, r2, r3, ..., r60] -> predict r61

    Returns:
        X shape = (samples, lookback, 1)
        y shape = (samples,)
    """

    values = np.asarray(values).reshape(-1, 1)

    X = []
    y = []

    for i in range(lookback, len(values)):
        X.append(values[i - lookback : i, 0])
        y.append(values[i, 0])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if len(X) == 0:
        raise ValueError(
            f"Not enough observations to create sequences with LOOKBACK={lookback}."
        )

    X = X.reshape(-1, lookback, 1)

    return X, y


# ===============================================================
# 4. MODEL
# ===============================================================


def build_model(lookback):
    """
    Build the LSTM return-prediction model.
    """

    model = Sequential(
        [
            Input(shape=(lookback, 1)),
            LSTM(64, return_sequences=True),
            Dropout(0.20),
            LSTM(32, return_sequences=False),
            Dropout(0.20),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse")

    return model


# ===============================================================
# 5. TRAIN MODEL
# ===============================================================


def train_model(returns, lookback, epochs, batch_size, verbose=0):
    """
    Fit scaler ONLY on the supplied historical returns.

    The caller controls what historical period is supplied,
    preventing future information from leaking into the model.
    """

    returns = np.asarray(returns, dtype=float)

    if len(returns) <= lookback + 10:
        raise ValueError("Not enough historical returns to train the model.")

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_returns = scaler.fit_transform(returns.reshape(-1, 1))

    X, y = make_sequences(scaled_returns, lookback)

    # Chronological validation split.
    split = int(len(X) * 0.90)

    if split <= 0 or split >= len(X):
        raise ValueError("Unable to create chronological validation split.")

    X_train = X[:split]
    X_val = X[split:]

    y_train = y[:split]
    y_val = y[split:]

    model = build_model(lookback)

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        shuffle=False,
        verbose=verbose,
    )

    return model, scaler


# ===============================================================
# 6. ONE-STEP PREDICTION
# ===============================================================


def predict_next_return(model, scaler, historical_returns, lookback):
    """
    Predict the next return using only historical returns.
    """

    window = np.asarray(historical_returns[-lookback:], dtype=float).reshape(-1, 1)

    scaled_window = scaler.transform(window).reshape(1, lookback, 1)

    prediction_scaled = model.predict(scaled_window, verbose=0)[0, 0]

    prediction_return = scaler.inverse_transform(np.array([[prediction_scaled]]))[0, 0]

    return float(prediction_return)


# ===============================================================
# 7. WALK-FORWARD BACKTEST
# ===============================================================


def walk_forward_backtest(df, lookback, test_days, retrain_every):
    """
    Expanding-window walk-forward backtest.

    At each prediction point:

        past data
             ↓
        fit scaler
             ↓
        train model
             ↓
        predict NEXT return
             ↓
        reveal actual return
             ↓
        expand historical dataset
             ↓
        repeat

    No future test observation is used when predicting that
    observation.

    The model is retrained every `retrain_every` observations to
    keep runtime manageable.
    """

    closes = df["Close"].values.astype(float)
    returns = df["Return"].values.astype(float)

    n = len(df)

    test_start = n - test_days

    if test_start <= lookback + 20:
        raise ValueError("Not enough history for the requested walk-forward test.")

    predictions = []
    actual_returns = []
    actual_prices = []
    previous_prices = []
    prediction_dates = []

    model = None
    scaler = None

    print("\n" + "=" * 65)
    print("WALK-FORWARD BACKTEST")
    print("=" * 65)

    for i in range(test_start, n):
        # Train using information available BEFORE day i.
        historical_returns = returns[:i]

        should_retrain = model is None or (i - test_start) % retrain_every == 0

        if should_retrain:
            print(
                f"Training model for test observation "
                f"{i - test_start + 1}/{test_days}..."
            )

            model, scaler = train_model(
                historical_returns, lookback, WF_EPOCHS, BATCH_SIZE, verbose=0
            )

        predicted_return = predict_next_return(
            model, scaler, historical_returns, lookback
        )

        previous_close = closes[i - 1]

        predicted_price = previous_close * (1.0 + predicted_return)

        actual_return = returns[i]

        predictions.append(predicted_price)
        actual_returns.append(actual_return)
        actual_prices.append(closes[i])
        previous_prices.append(previous_close)
        prediction_dates.append(df["Date"].iloc[i])

    predictions = np.asarray(predictions)
    actual_prices = np.asarray(actual_prices)
    previous_prices = np.asarray(previous_prices)
    actual_returns = np.asarray(actual_returns)

    # -----------------------------------------------------------
    # LSTM metrics
    # -----------------------------------------------------------

    rmse = np.sqrt(mean_squared_error(actual_prices, predictions))

    mae = mean_absolute_error(actual_prices, predictions)

    # Avoid division by zero.
    nonzero = actual_prices != 0

    mape = (
        np.mean(
            np.abs(
                (actual_prices[nonzero] - predictions[nonzero]) / actual_prices[nonzero]
            )
        )
        * 100
    )

    actual_direction = np.sign(actual_prices - previous_prices)

    predicted_direction = np.sign(predictions - previous_prices)

    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100

    # -----------------------------------------------------------
    # Naive baseline
    # -----------------------------------------------------------

    naive_predictions = previous_prices

    naive_rmse = np.sqrt(mean_squared_error(actual_prices, naive_predictions))

    naive_mae = mean_absolute_error(actual_prices, naive_predictions)

    naive_mape = (
        np.mean(
            np.abs(
                (actual_prices[nonzero] - naive_predictions[nonzero])
                / actual_prices[nonzero]
            )
        )
        * 100
    )

    naive_direction = np.sign(naive_predictions - previous_prices)

    # Since naive prediction is exactly previous price,
    # its directional prediction is neutral.
    # Therefore, a fair directional baseline is simply
    # the historical proportion of positive/negative moves.
    baseline_directional_accuracy = (
        max(np.mean(actual_direction > 0), np.mean(actual_direction < 0)) * 100
    )

    # -----------------------------------------------------------
    # Print results
    # -----------------------------------------------------------

    print("\n--- LSTM performance ---")

    print(f"RMSE:                 {rmse:.4f}")

    print(f"MAE:                  {mae:.4f}")

    print(f"MAPE:                 {mape:.2f}%")

    print(f"Directional accuracy: {directional_accuracy:.1f}%")

    print("\n--- Naive baseline: tomorrow = today ---")

    print(f"RMSE:                 {naive_rmse:.4f}")

    print(f"MAE:                  {naive_mae:.4f}")

    print(f"MAPE:                 {naive_mape:.2f}%")

    print(f"Directional baseline: {baseline_directional_accuracy:.1f}%")

    print("\n--- LSTM vs Naive ---")

    mae_improvement = (naive_mae - mae) / naive_mae * 100

    rmse_improvement = (naive_rmse - rmse) / naive_rmse * 100

    print(f"MAE improvement:  {mae_improvement:+.2f}%")

    print(f"RMSE improvement: {rmse_improvement:+.2f}%")

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "dir_acc": directional_accuracy,
        "naive_rmse": naive_rmse,
        "naive_mae": naive_mae,
        "naive_mape": naive_mape,
        "baseline_dir_acc": baseline_directional_accuracy,
        "mae_improvement": mae_improvement,
        "rmse_improvement": rmse_improvement,
        "preds": predictions,
        "actuals": actual_prices,
        "previous_prices": previous_prices,
        "dates": prediction_dates,
        "actual_returns": actual_returns,
    }


# ===============================================================
# 8. FINAL MODEL
# ===============================================================


def train_final_model(df, lookback):
    """
    Train the final forecasting model using ALL available
    historical returns.

    Since we're forecasting the future, there is no future
    leakage here.
    """

    returns = df["Return"].values.astype(float)

    print("\n" + "=" * 65)
    print("TRAINING FINAL MODEL")
    print("=" * 65)

    model, scaler = train_model(returns, lookback, EPOCHS, BATCH_SIZE, verbose=1)

    return model, scaler


# ===============================================================
# 9. MONTE-CARLO RECURSIVE FORECAST
# ===============================================================


def recursive_forecast_mc(
    model, scaler, historical_returns, last_close, lookback, horizon, mc_passes
):
    """
    Recursive return forecast with Monte-Carlo dropout.

    Each simulated path predicts:

        return(t+1)
             ↓
        price(t+1)
             ↓
        return(t+2)
             ↓
        price(t+2)
             ↓
        ...

    Dropout remains active during prediction, creating multiple
    stochastic model paths.

    The resulting P10/P90 values are empirical MC-dropout
    percentiles, NOT guaranteed statistical confidence intervals.
    """

    historical_returns = np.asarray(historical_returns, dtype=float)

    scaled_seed = scaler.transform(
        historical_returns[-lookback:].reshape(-1, 1)
    ).flatten()

    windows = np.tile(scaled_seed, (mc_passes, 1))

    current_prices = np.full(mc_passes, last_close, dtype=float)

    forecast_prices = np.zeros((mc_passes, horizon), dtype=float)

    forecast_returns = np.zeros((mc_passes, horizon), dtype=float)

    for day in range(horizon):
        x = windows[:, -lookback:].reshape(mc_passes, lookback, 1)

        # training=True keeps dropout active.
        predicted_scaled_returns = model(x, training=True).numpy().flatten()

        predicted_returns = scaler.inverse_transform(
            predicted_scaled_returns.reshape(-1, 1)
        ).flatten()

        # Prevent mathematically nonsensical prices.
        predicted_returns = np.clip(predicted_returns, -0.50, 0.50)

        new_prices = current_prices * (1.0 + predicted_returns)

        forecast_returns[:, day] = predicted_returns
        forecast_prices[:, day] = new_prices

        current_prices = new_prices

        windows = np.concatenate(
            [windows, predicted_scaled_returns.reshape(-1, 1)], axis=1
        )

    return forecast_prices, forecast_returns


# ===============================================================
# 10. MODEL RELIABILITY
# ===============================================================


def assess_reliability(backtest):
    """
    Give a simple qualitative assessment.

    This is deliberately conservative.
    """

    score = 0

    # LSTM must beat naive MAE.
    if backtest["mae"] < backtest["naive_mae"]:
        score += 1

    # LSTM must beat naive RMSE.
    if backtest["rmse"] < backtest["naive_rmse"]:
        score += 1

    # Directional accuracy above 50%.
    if backtest["dir_acc"] >= 55:
        score += 1

    # MAPE should be reasonably low.
    if backtest["mape"] <= 3:
        score += 1

    if score >= 4:
        return "HIGH"

    if score >= 2:
        return "MEDIUM"

    return "LOW"


# ===============================================================
# 11. SIGNAL
# ===============================================================


def generate_signal(expected_return, probability_positive, reliability):
    """
    Conservative signal generation.

    BUY requires:
        - expected return above threshold
        - probability of positive outcome >= threshold
        - at least MEDIUM historical reliability

    Otherwise HOLD / AVOID.
    """

    if (
        expected_return >= BUY_THRESHOLD
        and probability_positive >= MIN_PROBABILITY_POSITIVE
        and reliability in {"MEDIUM", "HIGH"}
    ):
        return "BUY"

    return "HOLD / AVOID"


# ===============================================================
# 12. FORECAST PLOT
# ===============================================================


def plot_forecast(df, mc_paths, horizon):
    """
    Plot historical prices + forecast mean + MC percentile band.
    """

    hist_window = min(120, len(df))

    historical_dates = df["Date"].iloc[-hist_window:]
    historical_prices = df["Close"].iloc[-hist_window:]

    last_date = df["Date"].iloc[-1]

    # These are business days, not guaranteed NSE trading days.
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=horizon
    )

    mean_path = mc_paths.mean(axis=0)

    lower = np.percentile(mc_paths, 10, axis=0)

    upper = np.percentile(mc_paths, 90, axis=0)

    plt.figure(figsize=(14, 6))

    plt.plot(historical_dates, historical_prices, label="Historical Close")

    plt.plot(future_dates, mean_path, label="Forecast Mean")

    plt.fill_between(future_dates, lower, upper, alpha=0.20, label="MC Dropout P10-P90")

    plt.axvline(last_date, linestyle="--", alpha=0.6, label="Forecast Start")

    plt.title("LSTM 30-Day Recursive Forecast")

    plt.xlabel("Date")

    plt.ylabel("Close Price (INR)")

    plt.legend()

    plt.grid(True, alpha=0.25)

    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT, dpi=150)

    plt.show()


# ===============================================================
# 13. MAIN
# ===============================================================


def main():

    print("=" * 65)
    print("LSTM STOCK RETURN FORECASTER")
    print("=" * 65)

    # -----------------------------------------------------------
    # Load
    # -----------------------------------------------------------

    df = load_close_series(CSV_PATH)

    print(f"\nLoaded {len(df)} price observations.")

    print(f"Date range: {df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}")

    # -----------------------------------------------------------
    # Returns
    # -----------------------------------------------------------

    df = add_returns(df)

    close_values = df["Close"].values.astype(float)

    if len(close_values) <= LOOKBACK + TEST_DAYS:
        raise ValueError(
            f"Not enough history.\n"
            f"Need substantially more than "
            f"{LOOKBACK + TEST_DAYS} observations."
        )

    # -----------------------------------------------------------
    # Walk-forward backtest
    # -----------------------------------------------------------

    backtest = walk_forward_backtest(df, LOOKBACK, TEST_DAYS, RETRAIN_EVERY)

    reliability = assess_reliability(backtest)

    print(f"\nModel reliability: {reliability}")

    # -----------------------------------------------------------
    # Train final model on ALL historical data
    # -----------------------------------------------------------

    model, scaler = train_final_model(df, LOOKBACK)

    # -----------------------------------------------------------
    # 30-day Monte-Carlo forecast
    # -----------------------------------------------------------

    historical_returns = df["Return"].values.astype(float)

    last_actual = close_values[-1]

    mc_paths, mc_returns = recursive_forecast_mc(
        model=model,
        scaler=scaler,
        historical_returns=historical_returns,
        last_close=last_actual,
        lookback=LOOKBACK,
        horizon=FORECAST_HORIZON,
        mc_passes=MC_PASSES,
    )

    # -----------------------------------------------------------
    # Day 30 statistics
    # -----------------------------------------------------------

    day30 = mc_paths[:, -1]

    mean_day30 = np.mean(day30)

    p10 = np.percentile(day30, 10)

    p90 = np.percentile(day30, 90)

    expected_return = (mean_day30 - last_actual) / last_actual * 100

    p10_return = (p10 - last_actual) / last_actual * 100

    p90_return = (p90 - last_actual) / last_actual * 100

    probability_positive = np.mean(day30 > last_actual) * 100

    probability_negative = np.mean(day30 < last_actual) * 100

    # -----------------------------------------------------------
    # Signal
    # -----------------------------------------------------------

    signal = generate_signal(expected_return, probability_positive, reliability)

    # -----------------------------------------------------------
    # Final report
    # -----------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("30-DAY FORECAST")
    print("=" * 65)

    print(f"Last actual close:              ₹{last_actual:.2f}")

    print(f"Day-30 mean forecast:            ₹{mean_day30:.2f}")

    print(f"Expected 30-day return:          {expected_return:+.2f}%")

    print(f"MC P10 forecast:                 ₹{p10:.2f}")

    print(f"MC P10 return:                   {p10_return:+.2f}%")

    print(f"MC P90 forecast:                 ₹{p90:.2f}")

    print(f"MC P90 return:                   {p90_return:+.2f}%")

    print(f"Probability of positive outcome: {probability_positive:.1f}%")

    print(f"Probability of negative outcome: {probability_negative:.1f}%")

    print(
        "\nNote: P10/P90 are empirical Monte-Carlo dropout "
        "percentiles, not guaranteed confidence intervals."
    )

    # -----------------------------------------------------------
    # Signal report
    # -----------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("MODEL SIGNAL")
    print("=" * 65)

    print(f"BUY threshold:                  +{BUY_THRESHOLD:.1f}%")

    print(f"Required positive probability:   {MIN_PROBABILITY_POSITIVE:.1f}%")

    print(f"Historical model reliability:    {reliability}")

    print(f"FINAL SIGNAL:                    {signal}")

    # -----------------------------------------------------------
    # Caveats
    # -----------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("INTERPRETATION")
    print("=" * 65)

    if backtest["mae"] < backtest["naive_mae"]:
        print("✓ LSTM beat the naive baseline on MAE.")
    else:
        print("✗ LSTM did NOT beat the naive baseline on MAE.")

    if backtest["rmse"] < backtest["naive_rmse"]:
        print("✓ LSTM beat the naive baseline on RMSE.")
    else:
        print("✗ LSTM did NOT beat the naive baseline on RMSE.")

    if backtest["dir_acc"] >= 55:
        print(f"✓ Directional accuracy was {backtest['dir_acc']:.1f}%.")
    else:
        print(f"✗ Directional accuracy was only {backtest['dir_acc']:.1f}%.")

    if expected_return >= BUY_THRESHOLD:
        print(f"✓ Expected return exceeds the +{BUY_THRESHOLD:.1f}% threshold.")
    else:
        print(f"✗ Expected return does not exceed the +{BUY_THRESHOLD:.1f}% threshold.")

    if probability_positive >= MIN_PROBABILITY_POSITIVE:
        print(f"✓ Positive-outcome probability is {probability_positive:.1f}%.")
    else:
        print(f"✗ Positive-outcome probability is only {probability_positive:.1f}%.")

    if reliability == "LOW":
        print(
            "\nWARNING: Historical model performance is weak. "
            "The 30-day forecast should not be treated as a "
            "reliable trading signal."
        )

    # -----------------------------------------------------------
    # Plot
    # -----------------------------------------------------------

    plot_forecast(df, mc_paths, FORECAST_HORIZON)

    print(f"\nForecast plot saved to: {OUTPUT_PLOT}")


# ===============================================================
# ENTRY POINT
# ===============================================================

if __name__ == "__main__":
    main()
