"""End-to-end pipeline: peer-cohort scoring, classifier, explainability.

Point this at any tabular dataset with a numeric column worth screening
for anomalies and categorical columns that define a sensible peer
group (region, category, type, whatever makes two rows comparable).
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from .classifier import best_threshold, fit_classifier, sweep_thresholds, tune_random_forest
from .scoring import flag_candidates, score_peer_anomalies


def run_pipeline(
    df,
    value_col,
    group_cols,
    additional_feature_cols,
    min_cohort_size=5,
    z_threshold=2.0,
    test_size=0.2,
    random_state=42,
):
    """Run the full pipeline and return the fitted artifacts and
    evaluation results in one dict, ready for inspection or reuse.

    Parameters
    ----------
    df : DataFrame
        Already free of missing values in value_col, group_cols, and
        additional_feature_cols, drop or impute before calling this.
    value_col : str
        Numeric column to screen for peer-anomalies.
    group_cols : list[str]
        Columns defining each row's peer cohort.
    additional_feature_cols : list[str]
        Extra columns to feed the classifier alongside value_col and
        the cohort size (added automatically). Do not include any
        column derived from value_col's own z-score or flag, that
        would leak the label into the features.
    """
    scored = score_peer_anomalies(df, value_col, group_cols, min_cohort_size=min_cohort_size)
    scored = flag_candidates(scored, f"{value_col}_loo_zscore", threshold=z_threshold)
    label_col = f"{value_col}_loo_zscore_flag"

    labeled = scored.dropna(subset=[label_col]).copy()
    labeled[label_col] = labeled[label_col].astype(bool)

    feature_cols = list(additional_feature_cols) + [value_col, f"{value_col}_cohort_size"]
    X = pd.get_dummies(labeled[feature_cols])
    y = labeled[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    best_params, best_cv_f1 = tune_random_forest(X, y, random_state=random_state)
    model = fit_classifier(X_train, y_train, best_params, random_state=random_state)
    sweep = sweep_thresholds(model, X_test, y_test)
    chosen_threshold = best_threshold(sweep, metric="f1")

    return {
        "scored_data": scored,
        "model": model,
        "best_params": best_params,
        "best_cv_f1": best_cv_f1,
        "threshold_sweep": sweep,
        "chosen_threshold": chosen_threshold,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_columns": list(X.columns),
        "label_column": label_col,
    }
