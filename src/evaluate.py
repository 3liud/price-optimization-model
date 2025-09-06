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
    """
    Load data and model artifacts for evaluation.

    Returns:
        df (pd.DataFrame): preprocessed daily data.
        pipe (Pipeline): trained demand model.
        features (list): features used in the model.
    """
    df = pd.read_parquet(Path(PROC_DIR) / "daily.parquet")
    bundle = joblib.load(Path(MODEL_DIR) / "demand_model.pkl")
    pipe = bundle["pipe"]
    features = bundle.get("features", ["log_price"] + FEATURES_CTX)
    return df, pipe, features


def build_holdout(df: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """
    Split the data into a holdout set for evaluation. The holdout set starts `weeks`
    before the last date in the data and goes until the end. Additionally, compute
    the median price per SKU during the training period (<= cutoff) as a baseline
    price for evaluation. Also, compute the median price per SKU over all history
    as a proxy for unit costs.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed daily data.
    weeks : int
        Number of weeks to reserve at the end of the data for evaluation.

    Returns
    -------
    pd.DataFrame
        The holdout set with additional columns for baseline price and median price
        per SKU.
    """
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
    """
    Attach recommended prices from a CSV file to the holdout set if present. If
    the file does not exist or is empty, fall back to the current price.

    Parameters
    ----------
    hold : pd.DataFrame
        The holdout set with additional columns for baseline price and median price
        per SKU.

    Returns
    -------
    pd.DataFrame
        The holdout set with additional column for recommended price per SKU.
    """
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
    """
    Predict demand quantities using the trained demand model.

    Parameters
    ----------
    pipe : Pipeline
        Trained demand model (HistGradientBoostingRegressor).
    frame : pd.DataFrame
        DataFrame with features and context.
    price_series : pd.Series
        Series of prices to predict demand for.

    Returns
    -------
    np.ndarray
        Predicted demand quantities.
    """
    X = frame[FEATURES_CTX].copy()
    X.insert(0, "log_price", np.log(np.clip(price_series.values, 1e-6, None)))
    log_qty = pipe.predict(X[["log_price"] + FEATURES_CTX])
    return np.expm1(log_qty).clip(min=0)


def main():
    """
    Evaluate the profit uplift of the optimized prices compared to the baseline (median historical price).

    Saves a summary JSON with the following keys:
        - holdout_days: number of days in the holdout period
        - rows: number of rows in the holdout DataFrame
        - baseline_profit_total: sum of profit under the baseline prices
        - recommended_profit_total: sum of profit under the recommended prices
        - profit_uplift: difference in profit between recommended and baseline
        - uplift_pct_vs_baseline: profit uplift as a percentage of the baseline profit

    Also saves a sample CSV of the uplift evaluation with the following columns:
        - date: date of the observation
        - StockCode: SKU identifier
        - Country_top: top-level country for the SKU
        - price: current price of the SKU
        - baseline_price: median historical price of the SKU
        - recommended_price: optimized price of the SKU
        - pred_qty_baseline: predicted quantity under the baseline price
        - pred_qty_reco: predicted quantity under the recommended price
        - base_profit: profit under the baseline price
        - reco_profit: profit under the recommended price
    """
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
