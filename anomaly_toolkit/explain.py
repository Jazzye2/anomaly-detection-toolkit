"""Per-prediction explainability with SHAP.

A Random Forest wins on raw performance but, as an ensemble of hundreds
of trees, can't be read in plain language the way one small decision
tree can. SHAP closes that gap: it attributes each individual
prediction back to the features that drove it, so a flagged case can
still be explained to a reviewer without giving up the stronger model.
"""
import shap


def _as_float(X):
    """Cast every column to float64 before handing data to SHAP.

    get_dummies output and pandas' newer dtype backends (nullable and
    Arrow-backed types, the default in pandas >= 3.0) aren't always the
    plain numpy dtypes shap's compiled tree-walking code expects, so
    cast explicitly rather than rely on an implicit, version-dependent
    conversion.
    """
    return X.astype("float64")


def build_explainer(model, background_X):
    """TreeExplainer is exact and fast for tree ensembles like Random
    Forest, no need for the slower, approximate model-agnostic
    KernelExplainer that generic black-box models require.
    """
    return shap.TreeExplainer(model, _as_float(background_X))


def _positive_class_values(shap_values):
    """Normalize across shap versions: older versions return a list of
    one (n_samples, n_features) array per class; newer versions return
    a single (n_samples, n_features, n_classes) array instead.
    """
    if isinstance(shap_values, list):
        return shap_values[1]
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def explain_row(explainer, row_X, top_n=5):
    """Return the top_n features driving one row's prediction, as
    (feature_name, shap_value) pairs sorted by absolute impact.
    A positive value pushed the prediction toward "flagged"; negative
    pushed it toward "normal."
    """
    shap_values = _positive_class_values(
        explainer.shap_values(_as_float(row_X), check_additivity=False)
    )
    values = shap_values[0]
    pairs = list(zip(row_X.columns, values))
    pairs.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return pairs[:top_n]


def summary_plot(explainer, X_sample, out_path):
    """Save a SHAP summary plot: which features matter most, overall,
    across a sample of rows, not just one prediction.
    """
    import matplotlib.pyplot as plt

    X_sample = _as_float(X_sample)
    shap_values = _positive_class_values(
        explainer.shap_values(X_sample, check_additivity=False)
    )
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
