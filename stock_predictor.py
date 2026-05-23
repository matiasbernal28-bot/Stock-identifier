import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error


# ============================================================
# STEP 1 — Download stock data from Yahoo Finance
# We grab 5 years of weekly prices so the model has enough
# history to learn from. The "target" column is what we're
# trying to predict: how much will the price change next week?
# ============================================================

def get_stock_data(ticker, period="5y"):
    df = yf.download(ticker, period=period, interval="1wk", progress=False)

    if df.empty:
        return None

    # yfinance sometimes returns columns as a multi-level table — flatten it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # We predict % change instead of raw price because a $5 move on a $10
    # stock is very different from a $5 move on a $500 stock
    df["target"] = df["Close"].pct_change().shift(-1)

    # Keep the last row even though it has no target yet —
    # it holds the real current price we show on the site
    return df


# ============================================================
# STEP 2 — Build signals the model can learn from
# Raw price data alone isn't enough. We calculate indicators
# that describe what the stock has been doing recently.
# ============================================================

def compute_rsi(prices, period=14):
    """
    RSI (Relative Strength Index) — measures if a stock is
    overbought (above 70) or oversold (below 30).
    Think of it as a momentum speedometer.
    """
    change = prices.diff()
    gains  = change.clip(lower=0)   # only positive days
    losses = -change.clip(upper=0)  # only negative days (flipped positive)

    avg_gain = gains.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = losses.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df):
    """
    Takes raw price data and adds 7 signals the model uses
    to make its prediction. Each one captures a different
    aspect of how the stock is behaving right now.
    """
    df = df.copy()

    # How much did the price move this week?
    df["weekly_return"] = df["Close"].pct_change()

    # Is trading volume unusually high compared to the last 4 weeks?
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(4).mean()

    # How wide was the price range this week? (high vs low)
    df["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]

    # Momentum — is the stock in an uptrend or downtrend?
    df["RSI"]    = compute_rsi(df["Close"])
    df["mom_4w"] = df["Close"].pct_change(4)   # last 4 weeks
    df["mom_8w"] = df["Close"].pct_change(8)   # last 8 weeks

    # How volatile has the stock been lately?
    df["volatility_4w"] = df["weekly_return"].rolling(4).std()

    # Keep the last row so we always have the real current price
    return df


# The 7 signals we feed into the model
FEATURES = [
    "weekly_return",
    "volume_ratio",
    "high_low_range",
    "RSI",
    "mom_4w",
    "mom_8w",
    "volatility_4w",
]


# ============================================================
# STEP 3 — Train the prediction model
# We use a Gradient Boosting model — it builds ~200-400 small
# decision trees, each one learning from the mistakes of the
# last. GridSearchCV automatically finds the best settings.
# ============================================================

# Settings we try during the search for the best model
SETTINGS_TO_TRY = {
    "max_iter":         [200, 400],       # number of trees
    "learning_rate":    [0.02, 0.05, 0.1],# how fast it learns
    "max_depth":        [3, 5],           # how deep each tree goes
    "min_samples_leaf": [10, 20],         # minimum data points per leaf
}


def train_model(df):
    # Only keep rows where we have both features AND a known target
    # (the last row has no target yet since we don't know next week)
    clean_df = df[FEATURES + ["target", "Close"]].dropna()

    X      = clean_df[FEATURES]  # inputs  (the 7 signals)
    y      = clean_df["target"]  # output  (next week's % change)
    closes = clean_df["Close"]   # needed later to turn % change back into a price

    # TimeSeriesSplit makes sure we always train on the past and test on the future
    time_split = TimeSeriesSplit(n_splits=5)

    # Try all combinations of settings and keep the best one
    search = GridSearchCV(
        HistGradientBoostingRegressor(random_state=42),
        SETTINGS_TO_TRY,
        cv=time_split,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    search.fit(X, y)

    # Re-train the winning settings on everything except the last 10 weeks
    # (those 10 weeks become our test to measure real accuracy)
    best_model = HistGradientBoostingRegressor(random_state=42, **search.best_params_)
    best_model.fit(X.iloc[:-10], y.iloc[:-10])

    # Measure accuracy by comparing predicted prices vs actual prices
    predicted_returns = best_model.predict(X.iloc[-10:])
    predicted_prices  = closes.iloc[-10:].values * (1 + predicted_returns)
    actual_prices     = closes.iloc[-10:].values * (1 + y.iloc[-10:].values)
    accuracy = mean_absolute_percentage_error(actual_prices, predicted_prices)

    return best_model, accuracy


# ============================================================
# STEP 4 — Make a prediction for next week
# Feed this week's signals into the trained model and convert
# the predicted % return back into a dollar price.
# ============================================================

def predict_next_week(model, df):
    this_week = df[FEATURES].iloc[[-1]]       # current week's signals
    predicted_return = model.predict(this_week)[0]
    current_price    = float(df["Close"].iloc[-1])
    return current_price * (1 + predicted_return)


# ============================================================
# STEP 5 — Calculate a risk score (0 to 100)
# This combines four danger signals into one easy number.
# Higher = more volatile / riskier stock right now.
# ============================================================

def compute_risk_score(df, predicted_price):
    current_price    = float(df["Close"].iloc[-1])
    predicted_change = (predicted_price - current_price) / current_price

    volatility    = float(df["volatility_4w"].iloc[-1])
    rsi           = float(df["RSI"].iloc[-1])
    weekly_range  = float(df["high_low_range"].iloc[-1])

    # Score each signal from 0 to 1, then combine them
    volatility_score = min(volatility / 0.10, 1.0)         # 10%+ weekly move = max danger
    rsi_score        = abs(rsi - 50) / 50                  # far from 50 = overbought or oversold
    range_score      = min(weekly_range / 0.15, 1.0)       # wide candle = unstable
    move_score       = min(abs(predicted_change) / 0.15, 1.0) # big predicted move = risky

    # Volatility matters most (35%), then RSI, range, and predicted move
    risk = (0.35 * volatility_score +
            0.25 * rsi_score        +
            0.20 * range_score      +
            0.20 * move_score)

    return round(risk * 100, 1), {
        "current_price":    round(current_price, 2),
        "predicted_price":  round(predicted_price, 2),
        "predicted_change": f"{predicted_change:+.2%}",
        "volatility_4w":    f"{volatility:.2%}",
        "RSI":              round(rsi, 1),
        "risk_score":       round(risk * 100, 1),
    }


# ============================================================
# HELPERS — Labels and recommendations
# ============================================================

USER_RISK_LEVELS = {
    "1": ("LOW",       0,  30),
    "2": ("MODERATE",  30, 50),
    "3": ("HIGH",      50, 70),
    "4": ("VERY HIGH", 70, 100),
}


def risk_label(score: float) -> str:
    if score >= 70: return "VERY HIGH"
    if score >= 50: return "HIGH"
    if score >= 30: return "MODERATE"
    return "LOW"


def get_recommendation(stock_score: float, user_tolerance: str) -> str:
    levels = ["LOW", "MODERATE", "HIGH", "VERY HIGH"]
    gap = levels.index(risk_label(stock_score)) - levels.index(user_tolerance)
    if gap <= 0: return "SUITABLE  — this stock fits within your risk tolerance."
    if gap == 1: return "CAUTION   — this stock is one level riskier than your tolerance."
    return "AVOID     — this stock significantly exceeds your risk tolerance."


def prompt_user_risk() -> str | None:
    print("\n  Your risk tolerance:")
    for key, (label, lo, hi) in USER_RISK_LEVELS.items():
        print(f"    [{key}] {label}  (scores {lo}–{hi}/100)")
    choice = input("  Enter 1-4 (or press Enter to skip): ").strip()
    if not choice:
        return None
    if choice not in USER_RISK_LEVELS:
        print("  Invalid choice — skipping.")
        return None
    return USER_RISK_LEVELS[choice][0]


# ============================================================
# SCREENER — Run the full watchlist and rank by risk
# ============================================================

DEFAULT_WATCHLIST = [
    # Big tech
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
    # High risk / speculative
    "MSTR", "COIN", "RIOT", "MARA", "HOOD",
    # Biotech (very volatile)
    "SAVA", "NKTR", "OCGN",
    # ETFs (used as benchmarks)
    "SPY", "QQQ", "ARKK",
]


def screen_tickers(tickers: list[str]) -> pd.DataFrame:
    results = []

    for ticker in tickers:
        print(f"  Analysing {ticker}...")
        raw = get_stock_data(ticker)

        if raw is None or len(raw) < 32:
            print(f"    Skipping {ticker} — not enough history")
            continue

        df = build_features(raw)

        if df[FEATURES].dropna().shape[0] < 20:
            continue

        try:
            model, accuracy = train_model(df)
            predicted_price = predict_next_week(model, df)
            risk_score, info = compute_risk_score(df, predicted_price)

            results.append({
                "ticker":   ticker,
                "accuracy": f"{accuracy:.2%} MAPE",
                **info,
            })
        except Exception as e:
            print(f"    Could not analyse {ticker}: {e}")

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("risk_score", ascending=False).reset_index(drop=True)


# ============================================================
# SINGLE TICKER LOOKUP
# ============================================================

def lookup_ticker(ticker: str, user_tolerance: str | None = None):
    ticker = ticker.strip().upper()
    print(f"\nLooking up {ticker}...")

    raw = get_stock_data(ticker)
    if raw is None or len(raw) < 32:
        print(f"  Couldn't find enough data for '{ticker}'. Double-check the symbol.")
        return

    df = build_features(raw)
    if df[FEATURES].dropna().shape[0] < 20:
        print(f"  Not enough history to make a prediction for '{ticker}'.")
        return

    try:
        model, accuracy = train_model(df)
        predicted_price = predict_next_week(model, df)
        risk_score, info = compute_risk_score(df, predicted_price)
    except Exception as e:
        print(f"  Something went wrong: {e}")
        return

    label = risk_label(risk_score)

    print(f"""
{'='*45}
  {ticker} — Weekly Risk Report
{'='*45}
  Current price    : ${info['current_price']}
  Predicted price  : ${info['predicted_price']}  ({info['predicted_change']})
  Risk score       : {risk_score}/100  [{label}]
{'─'*45}
  RSI              : {info['RSI']}
  Weekly volatility : {info['volatility_4w']}
  Model accuracy   : {accuracy:.2%} error (last 10 weeks)
{'='*45}""")

    if user_tolerance:
        rec = get_recommendation(risk_score, user_tolerance)
        print(f"  Your tolerance   : {user_tolerance}")
        print(f"  Recommendation   : {rec}")
        print(f"{'='*45}")

    print(f"  Risk score: 0 = low risk, 100 = extreme risk")
    print(f"  Prediction covers: next weekly close\n")


# ============================================================
# MAIN — Run the full screener
# ============================================================

def main(tickers=None, top_n=10):
    watchlist = tickers or DEFAULT_WATCHLIST
    print(f"\nWeekly Stock Risk Screener")
    print(f"Checking {len(watchlist)} stocks...\n")

    results = screen_tickers(watchlist)

    if results.empty:
        print("No results. Check your internet connection or ticker list.")
        return

    print(f"\n{'='*70}")
    print(f"  TOP {top_n} RISKIEST STOCKS THIS WEEK")
    print(f"{'='*70}")

    display = ["ticker", "risk_score", "current_price", "predicted_price",
               "predicted_change", "volatility_4w", "RSI", "accuracy"]
    print(results[display].head(top_n).to_string(index=False))
    print(f"\nRisk score: 0 = low risk, 100 = extreme risk")
    print(f"Prediction covers: next weekly close\n")

    results.to_csv("weekly_risk_report.csv", index=False)
    print(f"Full results saved to weekly_risk_report.csv")


# ============================================================
# INTERACTIVE MODE — Type a ticker, get results
# ============================================================

def interactive_mode():
    print("\nStock Risk Predictor")
    print("Type a ticker to look it up. Type 'screen' to rank all stocks. Type 'quit' to exit.\n")
    while True:
        user_input = input("Enter ticker: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "screen":
            main()
        else:
            tolerance = prompt_user_risk()
            lookup_ticker(user_input, user_tolerance=tolerance)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        lookup_ticker(sys.argv[1])
    else:
        interactive_mode()
