"""Reference example: run the toolkit against the hospital staffing
dataset used in WGU D797 JCN1, reproducing that session's full
methodology end to end as a single script.

This is the pattern to copy when pointing the toolkit at a new
dataset: pick value_col (the number to screen for anomalies),
group_cols (what makes two rows comparable), and
additional_feature_cols (context for the classifier).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from anomaly_toolkit import build_explainer, explain_row, run_pipeline, summary_plot

DATA_PATH = Path(r"C:\Users\Fred\Documents\WGU\D797\JCN1-task\Hospital_Staffing_dataset.csv")
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

VALUE_COL = "Productive Hours per Adjusted Patient Day"
GROUP_COLS = ["County Name", "Type of Control", "Hours Type"]
ADDITIONAL_FEATURE_COLS = ["Year", "County Name", "Type of Control", "Hours Type", "Productive Hours"]
REQUIRED_COLS = ["Facility Number", "Facility Name", "Begin Date", "End Date", "Type of Control", VALUE_COL]


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=REQUIRED_COLS)

    result = run_pipeline(
        df,
        value_col=VALUE_COL,
        group_cols=GROUP_COLS,
        additional_feature_cols=ADDITIONAL_FEATURE_COLS,
    )

    print("Best hyperparameters:", result["best_params"])
    print("Best cross-validated F1:", round(result["best_cv_f1"], 3))
    print()
    print("Threshold sweep:")
    print(result["threshold_sweep"].to_string(index=False))
    print()
    print("Chosen threshold (F1-optimal):", result["chosen_threshold"])

    OUT_DIR.mkdir(exist_ok=True)
    result["threshold_sweep"].to_csv(OUT_DIR / "threshold_sweep.csv", index=False)

    explainer = build_explainer(result["model"], result["X_train"])

    proba = result["model"].predict_proba(result["X_test"])[:, 1]
    top_idx = int(np.argmax(proba))
    top_row = result["X_test"].iloc[[top_idx]]
    print()
    print(f"Most anomalous test-set row (model confidence {proba[top_idx]:.3f}), SHAP explanation:")
    for feature, impact in explain_row(explainer, top_row):
        direction = "pushed toward FLAGGED" if impact > 0 else "pushed toward normal"
        print(f"  {feature}: {impact:+.3f} ({direction})")

    summary_plot(explainer, result["X_test"].sample(min(500, len(result["X_test"])), random_state=42),
                 OUT_DIR / "shap_summary.png")
    print()
    print(f"Saved outputs/threshold_sweep.csv and outputs/shap_summary.png")


if __name__ == "__main__":
    main()
