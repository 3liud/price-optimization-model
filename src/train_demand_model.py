# src/train_demand_model.py
# Train demand model on daily per-SKU features and save pipeline + holdout MAE.

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import joblib
from config import PROC_DIR, MODEL_DIR, SEED, HOLDOUT_WEEKS

MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "log_price",
    "rolling_qty_7",
    "rolling_price_7",
    "rolling_price_change_7",
    "dow",
    "month",
    "is_weekend",
    "Country_top",
]
TARGET = "log_qty"


def load_frame() -> pd.DataFrame:
    df = pd.read_parquet(Path(PROC_DIR) / "daily.parquet")
    # guard
    df = df.dropna(subset=[TARGET] + FEATURES).copy()
    return df


def time_split(df: pd.DataFrame, holdout_weeks: int):
    cutoff = df["date"].max() - np.timedelta64(7 * holdout_weeks, "D")
    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    return train, test


def build_pipeline(countries: list) -> Pipeline:
    num_cols = [
        "log_price",
        "rolling_qty_7",
        "rolling_price_7",
        "rolling_price_change_7",
        "dow",
        "month",
        "is_weekend",
    ]
    cat_cols = ["Country_top"]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", categories=[countries]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    reg = HistGradientBoostingRegressor(
        max_depth=6,
        max_leaf_nodes=64,
        learning_rate=0.06,
        l2_regularization=0.01,
        random_state=SEED,
    )

    pipe = Pipeline(
        [
            ("pre", pre),
            ("reg", reg),
        ]
    )
    return pipe


def main():
    df = load_frame()
    train, test = time_split(df, HOLDOUT_WEEKS)

    # lock category order for reproducibility
    countries = sorted(df["Country_top"].astype(str).unique().tolist())
    pipe = build_pipeline(countries)

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    out_path = Path(MODEL_DIR) / "demand_model.pkl"
    joblib.dump(
        {
            "pipe": pipe,
            "countries": countries,
            "features": FEATURES,
            "target": TARGET,
            "mae_log_qty_holdout": float(mae),
            "cutoff_date": str(
                df["date"].max() - np.timedelta64(7 * HOLDOUT_WEEKS, "D")
            ),
        },
        out_path,
    )
    print(
        f"[train] Saved {out_path} | Holdout MAE(log_qty)={mae:.4f} | Test rows={len(X_test):,}"
    )


if __name__ == "__main__":
    main()
