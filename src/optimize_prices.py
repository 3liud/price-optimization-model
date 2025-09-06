# src/optimize_prices.py
# Recommend profit-maximizing prices per SKU-country using a trained demand model.

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from config import (
    PROC_DIR,
    MODEL_DIR,
    REPORT_DIR,
    COST_RATIO,
    GRID_STEPS,
    PRICE_LOWER_MULT,
    PRICE_UPPER_MULT,
)

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
    daily = pd.read_parquet(Path(PROC_DIR) / "daily.parquet")
    sku_stats = pd.read_parquet(Path(PROC_DIR) / "sku_stats.parquet")
    bundle = joblib.load(Path(MODEL_DIR) / "demand_model.pkl")
    pipe = bundle["pipe"]
    return daily, sku_stats, pipe


def simulate_profit(
    pipe, ctx_row: pd.Series, price_grid: np.ndarray, unit_cost: float
) -> pd.DataFrame:
    # Build model matrix by cloning context across candidate prices and recomputing log_price
    X = pd.DataFrame([ctx_row[FEATURES_CTX].to_dict()] * len(price_grid))
    X.insert(0, "log_price", np.log(np.clip(price_grid, 1e-6, None)))
    # Predict log-qty then invert
    log_qty_pred = pipe.predict(X[["log_price"] + FEATURES_CTX])
    qty_pred = np.expm1(log_qty_pred).clip(min=0)
    profit = (price_grid - unit_cost) * qty_pred
    return pd.DataFrame(
        {"candidate_price": price_grid, "pred_qty": qty_pred, "pred_profit": profit}
    )


def build_price_grid(hist_min: float, hist_max: float) -> np.ndarray:
    p_min = max(hist_min * PRICE_LOWER_MULT, 1e-6)
    p_max = max(p_min, hist_max * PRICE_UPPER_MULT)
    return np.linspace(p_min, p_max, GRID_STEPS)


def main():
    daily, sku_stats, pipe = load_artifacts()

    # Optimization context: latest row per SKU-country
    latest = (
        daily.sort_values("date")
        .groupby(["StockCode", "Country_top"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    ).merge(sku_stats, on="StockCode", how="left")

    recs = []
    for _, row in latest.iterrows():
        # Price grid within historical bounds
        grid = build_price_grid(row["hist_min_price"], row["hist_max_price"])

        # Unit cost proxy from median price
        unit_cost = float(row["median_price"] * COST_RATIO)

        sim = simulate_profit(pipe, row, grid, unit_cost)
        best_idx = int(sim["pred_profit"].idxmax())
        best = sim.loc[best_idx]

        recs.append(
            {
                "StockCode": row["StockCode"],
                "Country_top": row["Country_top"],
                "recommended_price": float(best["candidate_price"]),
                "predicted_qty": float(best["pred_qty"]),
                "predicted_profit": float(best["pred_profit"]),
                "unit_cost_assumed": unit_cost,
                "hist_min_price": float(row["hist_min_price"]),
                "hist_max_price": float(row["hist_max_price"]),
                "median_price": float(row["median_price"]),
            }
        )

    recs_df = pd.DataFrame(recs).sort_values("predicted_profit", ascending=False)
    out_path = Path(REPORT_DIR) / "price_recos.csv"
    recs_df.to_csv(out_path, index=False)
    print(f"[optimize] Saved {out_path} | rows={len(recs_df):,}")


if __name__ == "__main__":
    main()
