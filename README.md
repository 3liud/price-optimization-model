# Price Optimization — Project Documentation

This project demonstrates a full **price optimization pipeline** using the UCI **Online Retail II** dataset. It covers data preparation, demand modeling, profit‑maximizing price recommendation, backtesting, and optional presentation in a Dash app.

---

## 1. Overview

The pipeline learns **demand response to price** at the SKU × Country × Day level. It then recommends profit‑maximizing prices by simulating candidate prices and predicting demand. Evaluation compares baseline (median historical) prices against optimized ones in a holdout period.

**Pipeline steps:**

1. **Data prep** → clean raw Excel sheets, aggregate, feature‑engineer.
2. **Train** → fit a demand model (HistGradientBoostingRegressor).
3. **Optimize** → simulate profit across candidate prices, choose max.
4. **Evaluate** → backtest uplift vs. baseline.

**Artifacts produced:**

* `data/processed/` → training data and SKU stats.
* `models/` → trained pipeline.
* `reports/` → recommendations and evaluation results.

---

## 2. Dataset

* **Name:** Online Retail II (UCI Machine Learning Repository)
* **Download:**

  * UCI overview: [https://archive.ics.uci.edu/ml/datasets/online+retail+ii](https://archive.ics.uci.edu/ml/datasets/online+retail+ii)
  * UCI dataset page: [https://archive.ics.uci.edu/dataset/502/online+retail+ii](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
  * Kaggle mirror (login required): [https://www.kaggle.com/datasets/nathaniel/uci-online-retail-ii-data-set/data](https://www.kaggle.com/datasets/nathaniel/uci-online-retail-ii-data-set/data)
* **What to do:** Download the Excel file and place it at:

```
data/raw/online_retail_ii.xlsx
```

* **Fields used:** Invoice, StockCode, Description, Quantity, InvoiceDate, Price/UnitPrice, CustomerID, Country.
* **Period:** 2009‑12‑01 → 2011‑12‑09.

---

## 3. Project Structure

```
project/
├── data/
│   ├── raw/online_retail_ii.xlsx
│   └── processed/{daily.parquet, sku_stats.parquet}
├── models/demand_model.pkl
├── reports/
│   ├── price_recos.csv
│   ├── eval_summary.json
│   └── uplift_sample.csv
├── src/
│   ├── config.py
│   ├── data_prep.py
│   ├── train_demand_model.py
│   ├── optimize_prices.py
│   └── evaluate.py
└── run_all.py
```

---

## 4. How to Run

From the project root (after installing dependencies like `pandas`, `scikit-learn`, `pyarrow`, `openpyxl`):

```bash
python run_all.py
```

This executes the full pipeline in order:

1. `src/data_prep.py`
2. `src/train_demand_model.py`
3. `src/optimize_prices.py`
4. `src/evaluate.py`

---

## 5. Outputs

### `data/processed/`

* **`daily.parquet`** → cleaned & feature‑engineered daily SKU dataset.
* **`sku_stats.parquet`** → per‑SKU historical min/max/median prices.

### `models/`

* **`demand_model.pkl`** → sklearn pipeline with preprocessing + model, plus metadata (features, holdout MAE).

### `reports/`

* **`price_recos.csv`** → per‑SKU recommended prices and predicted profits.
* **`eval_summary.json`** → holdout backtest totals & uplift percentage.
* **`uplift_sample.csv`** → row‑level baseline vs. recommended comparison.

---

## 6. Dash App (Optional)

You can build a lightweight **Dash app** on top of the artifacts to explore results interactively.

**Suggested tabs:**

* Overview (KPIs, summary)
* SKU Explorer (elasticity curves, recos)
* Recommendations (sortable DataTable)
* Backtest (uplift distribution)
* Admin (trigger pipeline steps)

This can live under `app/` and read directly from `reports/` and `models/`.

---

## 7. Configuration

`src/config.py` contains knobs to control the pipeline:

* `COST_RATIO` → assumed cost as fraction of median price.
* `GRID_STEPS` → how many candidate prices to simulate.
* `PRICE_LOWER_MULT`, `PRICE_UPPER_MULT` → bounds relative to historical.
* `MIN_OBS_PER_SKU` → filter sparse SKUs.
* `HOLDOUT_WEEKS` → length of evaluation period.

---

## 8. Next Steps / Extensions

* Replace proxy costs with actual unit costs per SKU.
* Add inventory or MAP constraints in optimization.
* Use LightGBM or CatBoost for richer categorical handling.
* Extend to multi‑objective optimization (profit + volume).
* Deploy the Dash app for stakeholder demos.

---

## 9. License & Attribution

Dataset: Online Retail II, UCI Machine Learning Repository.
Respect dataset licensing and terms of use.
