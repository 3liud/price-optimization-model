# src/data_prep.py
# Robust prep for UCI Online Retail II with safe types for Parquet

import pandas as pd
import numpy as np
from pathlib import Path
from config import (
    RAW_DATA,
    PROC_DIR,
    TOP_COUNTRIES,
    MIN_OBS_PER_SKU,
    MIN_PRICE,
    MIN_QTY,
)

PROC_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

RENAME_MAP = {
    "Invoice": "InvoiceNo",
    "Customer ID": "CustomerID",
    "Price": "UnitPrice",
}


def read_all_sheets(path: Path) -> pd.DataFrame:
    xl = pd.read_excel(path, sheet_name=None)
    if not xl:
        raise ValueError(f"No sheets found in {path}")
    frames = []
    for _, df in xl.items():
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns=RENAME_MAP)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out


def validate_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        raise ValueError(
            f"Missing required columns after normalization: {missing}\nFound: {sorted(cols)}"
        )

    df = df.copy()

    # Normalize key dtypes early to avoid parquet conversion issues
    df["StockCode"] = df["StockCode"].astype(str)
    df["Country"] = df["Country"].fillna("Unknown").astype(str)

    # Drop cancellations: invoices starting with 'C'
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[~df["InvoiceNo"].str.startswith("C")]

    # Keep only positive qty/price
    df = df[(df["Quantity"] >= MIN_QTY) & (df["UnitPrice"] >= MIN_PRICE)]

    # Timestamps and date
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["date"] = df["InvoiceDate"].dt.normalize()

    return df


def top_k_countries(df: pd.DataFrame, k: int) -> pd.DataFrame:
    top = df["Country"].value_counts().nlargest(k).index.tolist()
    df = df.copy()
    df["Country_top"] = np.where(
        df["Country"].isin(top), df["Country"], "Other"
    ).astype(str)
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    # Use sums to compute quantity-weighted price (avoid groupby.apply warnings)
    df = df.copy()
    df["value"] = df["UnitPrice"] * df["Quantity"]

    agg = df.groupby(["StockCode", "Country_top", "date"], as_index=False).agg(
        qty=("Quantity", "sum"), value=("value", "sum")
    )
    agg["price"] = (agg["value"] / agg["qty"]).replace([np.inf, -np.inf], np.nan)
    agg = agg.drop(columns="value")

    agg["date"] = pd.to_datetime(agg["date"])
    agg = agg.sort_values(["StockCode", "Country_top", "date"])

    # Calendar features
    agg["dow"] = agg["date"].dt.dayofweek
    agg["month"] = agg["date"].dt.month
    agg["is_weekend"] = agg["dow"].isin([5, 6]).astype(int)

    # Rolling features per SKU-country
    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date")
        g["rolling_qty_7"] = g["qty"].rolling(7, min_periods=1).mean()
        g["rolling_price_7"] = g["price"].rolling(7, min_periods=1).mean()
        g["rolling_price_change_7"] = (
            g["price"].pct_change().fillna(0).rolling(7, min_periods=1).mean().fillna(0)
        )
        return g

    agg = agg.groupby(["StockCode", "Country_top"], group_keys=False).apply(_roll)

    # Logs
    agg["log_qty"] = np.log1p(agg["qty"])
    agg["log_price"] = np.log(agg["price"].clip(lower=1e-6))

    # Ensure string dtypes for parquet safety
    agg["StockCode"] = agg["StockCode"].astype(str)
    agg["Country_top"] = agg["Country_top"].astype(str)
    return agg


def filter_sparse_skus(df: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    counts = df.groupby("StockCode")["date"].count()
    keep = counts[counts >= min_obs].index
    return df[df["StockCode"].isin(keep)].copy()


def persist_sku_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby("StockCode")
        .agg(
            hist_min_price=("price", "min"),
            hist_max_price=("price", "max"),
            median_price=("price", "median"),
        )
        .reset_index()
    )
    stats["StockCode"] = stats["StockCode"].astype(str)
    stats.to_parquet(Path(PROC_DIR) / "sku_stats.parquet", index=False)
    return stats


def main():
    df = read_all_sheets(Path(RAW_DATA))
    df = validate_and_standardize(df)
    df = top_k_countries(df, TOP_COUNTRIES)
    daily = aggregate_daily(df)
    daily = filter_sparse_skus(daily, MIN_OBS_PER_SKU)

    # final parquet-safe dtypes
    daily["StockCode"] = daily["StockCode"].astype(str)
    daily["Country_top"] = daily["Country_top"].astype(str)

    persist_sku_stats(daily)

    out_path = Path(PROC_DIR) / "daily.parquet"
    daily.to_parquet(out_path, index=False)
    print(
        f"[data_prep] Saved {out_path} | rows={len(daily):,} | skus={daily['StockCode'].nunique():,}"
    )


if __name__ == "__main__":
    main()
