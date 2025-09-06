# src/evaluate.py
# Out-of-time evaluation: compare baseline (median historical price) vs recommended price
# using the trained demand model to predict quantities, then compute profit uplift.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from config import PROC_DIR, MODEL_DIR, REPORT_DIR, HOLDOUT_WEEKS, COST_RATIO

Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)

FEATURES_CTX = [
    "rolling_qty_7",
    "rolling_price_7",
    "rolling_price_change_7",
    "dow",
    "month",
    "is_weekend",
    "Country_top",
]


def load_artifacts():
    df = pd.read_parquet(Path(PROC_DIR) / "daily.parquet")
    bundle = joblib.load(Path(MODEL_DIR) / "demand_model.pkl")
    pipe = bundle["pipe"]
    features = bundle.get("features", ["log_price"] + FEATURES_CTX)
    return df, pipe, features


def build_holdout(df: pd.DataFrame, weeks: int) -> pd.DataFrame:
    cutoff = df["date"].max() - np.timedelta64(7 * weeks, "D")
    hold = df[df["date"] > cutoff].copy()

    # Baseline price per SKU from *train* side (<= cutoff)
    base_price = (
        df[df["date"] <= cutoff]
        .groupby("StockCode")["price"]
        .median()
        .rename("baseline_price")
    )
    hold = hold.merge(base_price, on="StockCode", how="left")

    # Median price per SKU for unit-cost proxy (all history)
    med_price = df.groupby("StockCode")["price"].median().rename("median_price_all")
    hold = hold.merge(med_price, on="StockCode", how="left")
    return hold


def attach_recommendations(hold: pd.DataFrame) -> pd.DataFrame:
    recos_path = Path(REPORT_DIR) / "price_recos.csv"
    if not recos_path.exists():
        # No recommendations computed; fallback to current price
        hold["recommended_price"] = hold["price"]
        return hold

    recos = pd.read_csv(recos_path)
    if recos.empty:
        hold["recommended_price"] = hold["price"]
        return hold

    cols = ["StockCode", "Country_top", "recommended_price"]
    recos = recos[cols].drop_duplicates()
    hold = hold.merge(recos, on=["StockCode", "Country_top"], how="left")
    # Fallback to current if a given SKU-country has no reco
    hold["recommended_price"] = hold["recommended_price"].fillna(hold["price"])
    return hold


def predict_qty(pipe, frame: pd.DataFrame, price_series: pd.Series) -> np.ndarray:
    X = frame[FEATURES_CTX].copy()
    X.insert(0, "log_price", np.log(np.clip(price_series.values, 1e-6, None)))
    log_qty = pipe.predict(X[["log_price"] + FEATURES_CTX])
    return np.expm1(log_qty).clip(min=0)


def main():
    df, pipe, _features = load_artifacts()
    hold = build_holdout(df, HOLDOUT_WEEKS)
    hold = attach_recommendations(hold)

    # Baseline & Recommended demand predictions
    baseline_price = hold["baseline_price"].fillna(hold["price"])
    reco_price = hold["recommended_price"].fillna(hold["price"])

    qty_base = predict_qty(pipe, hold, baseline_price)
    qty_reco = predict_qty(pipe, hold, reco_price)

    # Unit cost proxy (SKU-level): median price * COST_RATIO
    unit_cost = (hold["median_price_all"] * COST_RATIO).fillna(0.0)

    base_profit = (baseline_price - unit_cost) * qty_base
    reco_profit = (reco_price - unit_cost) * qty_reco

    base_total = float(base_profit.sum())
    reco_total = float(reco_profit.sum())
    uplift = float(reco_total - base_total)
    uplift_pct = float((reco_total / base_total - 1.0) if base_total > 0 else np.nan)

    summary = {
        "holdout_days": int(hold["date"].nunique()),
        "rows": int(len(hold)),
        "baseline_profit_total": base_total,
        "recommended_profit_total": reco_total,
        "profit_uplift": uplift,
        "uplift_pct_vs_baseline": uplift_pct,
    }

    # Save artifacts
    eval_path = Path(REPORT_DIR) / "eval_summary.json"
    with open(eval_path, "w") as f:
        json.dump(summary, f, indent=2)

    sample_cols = [
        "date",
        "StockCode",
        "Country_top",
        "price",
        "baseline_price",
        "recommended_price",
    ]
    sample = hold[sample_cols].copy()
    sample["pred_qty_baseline"] = qty_base
    sample["pred_qty_reco"] = qty_reco
    sample["base_profit"] = base_profit
    sample["reco_profit"] = reco_profit
    sample_path = Path(REPORT_DIR) / "uplift_sample.csv"
    sample.to_csv(sample_path, index=False)

    print(f"[evaluate] Saved {eval_path} and {sample_path}")
    print("Summary:", summary)


if __name__ == "__main__":
    main()
