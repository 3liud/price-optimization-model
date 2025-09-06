# run_all.py
# Convenience runner to execute the full pipeline end-to-end from PROJECT ROOT.

import subprocess
import sys
from pathlib import Path

STEPS = [
    "src/data_prep.py",
    "src/train_demand_model.py",
    "src/optimize_prices.py",
    "src/evaluate.py",
]

def run(step: str) -> int:
    print(f"\n=== Running: {step} ===")
    return subprocess.call([sys.executable, step])

def main():
    project_root = Path(__file__).resolve().parent  # FIX: project root is the folder containing this file
    raw = project_root / "data" / "raw" / "online_retail_ii.xlsx"

    if not raw.exists():
        print(f"[error] Missing dataset: {raw}\n"
              f"Download the UCI Online Retail II Excel and save it at this path.")
        sys.exit(1)

    # Ensure expected output dirs exist (handles your 'model' vs 'models' mismatch gracefully)
    (project_root / "models").mkdir(parents=True, exist_ok=True)
    (project_root / "reports").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "processed").mkdir(parents=True, exist_ok=True)

    codes = []
    for step in STEPS:
        rc = run(str(project_root / step))
        codes.append((step, rc))
        if rc != 0:
            print(f"[stop] Step failed: {step} (exit {rc})")
            break

    print("\n=== Summary ===")
    for step, rc in codes:
        print(f"{step}: {'OK' if rc == 0 else f'EXIT {rc}'}")

if __name__ == "__main__":
    main()
