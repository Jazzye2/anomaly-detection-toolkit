"""Supervised classifier training and threshold selection.

Random Forest is the default and only model wired in here because it
was the validated winner across six algorithms tested against this
problem shape (peer-anomaly flags, ~3% positive rate, mixed categorical
and numeric features): see the WGU D797 JCN1 findings log for the full
comparison. Swap in a different estimator if a new dataset's shape
favors one.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score

DEFAULT_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 15, 20, 25],
    "max_features": ["sqrt", "log2", 0.3, 0.5],
    "min_samples_leaf": [1, 2, 5],
}


def tune_random_forest(X, y, param_grid=None, n_iter=10, cv_splits=5, n_jobs=2, random_state=42):
    """Randomized search for Random Forest hyperparameters, scored on F1.

    Searches max_features explicitly rather than leaving it at scikit-learn's
    default ("sqrt"), since that default under-explores the feature space on
    wide, heavily one-hot-encoded data and was the single biggest lever found
    when this was tuned against the hospital staffing dataset.

    n_jobs defaults to 2, not -1 (all cores), on purpose: each parallel
    worker holds its own copy of the forest and data, so full parallelism
    multiplies peak memory by the core count. That's fine on a local
    machine but can crash a memory-capped host (e.g. Streamlit Community
    Cloud's free tier, ~1GB). Raise n_jobs explicitly if running somewhere
    with memory to spare.
    """
    param_grid = param_grid or DEFAULT_PARAM_GRID
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=random_state),
        param_grid,
        n_iter=n_iter,
        scoring="f1",
        cv=skf,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    search.fit(X, y)
    return search.best_params_, search.best_score_


def fit_classifier(X_train, y_train, params, random_state=42):
    model = RandomForestClassifier(random_state=random_state, **params)
    model.fit(X_train, y_train)
    return model


def sweep_thresholds(model, X_test, y_test, thresholds=(0.5, 0.3, 0.2, 0.15, 0.1, 0.05)):
    """Precision, recall, and F1 at each candidate decision threshold.

    The default 0.5 cutoff has no special real-world meaning. The right
    threshold depends on the cost of a false positive versus a false
    negative for the actual use case, lower it to favor recall when
    missing a real case is the expensive mistake.
    """
    proba = model.predict_proba(X_test)[:, 1]
    rows = []
    for t in thresholds:
        pred = proba > t
        rows.append({
            "threshold": t,
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def best_threshold(sweep_df, metric="f1"):
    return sweep_df.loc[sweep_df[metric].idxmax(), "threshold"]
