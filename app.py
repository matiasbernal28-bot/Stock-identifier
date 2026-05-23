from flask import Flask, render_template, request, jsonify
from stock_predictor import (
    get_stock_data, build_features, train_model,
    predict_next_week, compute_risk_score, risk_label,
    get_recommendation, screen_tickers, DEFAULT_WATCHLIST,
)

app = Flask(__name__)

TOLERANCE_MAP = {
    "1": "LOW",
    "2": "MODERATE",
    "3": "HIGH",
    "4": "VERY HIGH",
    "low": "LOW",
    "moderate": "MODERATE",
    "high": "HIGH",
    "very high": "VERY HIGH",
}
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/lookup", methods=["POST"])
def api_lookup():
    data = request.get_json()
    ticker = (data.get("ticker") or "").strip().upper()
    tolerance_raw = (data.get("tolerance") or "").strip().lower()

    if not ticker:
        return jsonify({"error": "Ticker is required."}), 400
    tolerance = TOLERANCE_MAP.get(tolerance_raw)

    raw = get_stock_data(ticker)
    if raw is None or len(raw) < 32:
        return jsonify({"error": f"Not enough data for '{ticker}'. Check the symbol and try again."}), 404

    df = build_features(raw)
    if df.dropna().shape[0] < 20:
        return jsonify({"error": f"Not enough processed data for '{ticker}'."}), 404

    try:
        model, accuracy = train_model(df)
        pred = predict_next_week(model, df)
        risk_score, info = compute_risk_score(df, pred)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    label = risk_label(risk_score)
    recommendation = get_recommendation(risk_score, tolerance) if tolerance else None

    return jsonify({
        "ticker":           ticker,
        "current_price":    info["current_price"],
        "predicted_price":  info["predicted_price"],
        "predicted_change": info["predicted_change"],
        "risk_score":       risk_score,
        "risk_label":       label,
        "rsi":              info["RSI"],
        "volatility":       info["volatility_4w"],
        "val_mape":         f"{accuracy:.2%}",
        "tolerance":        tolerance,
        "recommendation":   recommendation,
    })


@app.route("/api/screen", methods=["GET"])
def api_screen():
    results = screen_tickers(DEFAULT_WATCHLIST)
    if results.empty:
        return jsonify([])
    records = results[[
        "ticker", "risk_score", "current_price", "predicted_price",
        "predicted_change", "volatility_4w", "RSI", "accuracy",
    ]].to_dict(orient="records")
    return jsonify(records)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
