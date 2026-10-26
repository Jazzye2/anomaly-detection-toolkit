"""Streamlit front end for the anomaly detection toolkit.

Built for a non-technical user: upload a CSV, pick a few columns from
dropdowns, click Run, get a sortable table of flagged rows with a
plain-language explanation for any one you select. No code, no
thresholds-as-numbers, no ROC curves.

Launch with: streamlit run app.py
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from anomaly_toolkit import build_explainer, explain_row, flag_candidates, score_peer_anomalies
from anomaly_toolkit.classifier import fit_classifier, tune_random_forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Hyperparameters validated against the hospital staffing dataset
# (see the WGU D797 JCN1 findings log for the full search). Used as the
# fast default; the "re-tune" option below re-searches for a new dataset.
VALIDATED_PARAMS = {"n_estimators": 200, "max_depth": 20, "max_features": 0.5, "min_samples_leaf": 1}

st.set_page_config(page_title="Anomaly Detection Toolkit", layout="wide")
st.title("Anomaly Detection Toolkit")
st.caption(
    "Peer-group anomaly scoring with an explainable classifier on top. "
    "Upload a dataset, tell it what to compare, and get a reviewable list of flags."
)

with st.sidebar:
    st.header("1. Data")
    uploaded = st.file_uploader("CSV file", type="csv")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        columns = list(df.columns)
        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]

        st.header("2. What to compare")
        identifier_col = st.selectbox(
            "Identifier column (for display only, e.g. a name or ID)", columns
        )
        value_col = st.selectbox(
            "Value to screen for anomalies (a number)", numeric_cols
        )
        group_cols = st.multiselect(
            "Peer group columns (what makes two rows comparable)",
            [c for c in columns if c not in (identifier_col, value_col)],
        )
        feature_cols = st.multiselect(
            "Extra context columns for the model",
            [c for c in columns if c not in (identifier_col, value_col)],
            default=group_cols,
        )

        st.header("3. Options")
        min_cohort = st.number_input("Minimum peer group size to score", min_value=2, value=5)
        retune = st.checkbox("Re-tune hyperparameters for this dataset (slower)", value=False)

        run_clicked = st.button("Run analysis", type="primary", disabled=not group_cols)
    else:
        run_clicked = False


@st.cache_data(show_spinner=False)
def _score(df, value_col, group_cols, min_cohort):
    scored = score_peer_anomalies(df, value_col, group_cols, min_cohort_size=min_cohort)
    scored = flag_candidates(scored, f"{value_col}_loo_zscore", threshold=2.0)
    return scored


@st.cache_data(show_spinner=False)
def _train(labeled, feature_cols, label_col, retune, random_state=42):
    X = pd.get_dummies(labeled[feature_cols])
    y = labeled[label_col].astype(bool)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    if retune:
        params, _ = tune_random_forest(X, y, random_state=random_state)
    else:
        params = VALIDATED_PARAMS
    model = fit_classifier(X_train, y_train, params, random_state=random_state)
    return model, X_train, X_test, y_test, params


if uploaded is None:
    st.info("Upload a CSV to get started.")
    st.session_state.pop("results_ready", None)
else:
    if run_clicked:
        value_col_label = f"{value_col}_loo_zscore"
        label_col = f"{value_col_label}_flag"

        with st.spinner("Scoring peer groups..."):
            scored = _score(df, value_col, group_cols, min_cohort)

        labeled = scored.dropna(subset=[label_col]).copy()
        missing_group_count = int(scored[f"{value_col}_missing_peer_group"].sum())
        too_few_peers_count = len(scored) - len(labeled) - missing_group_count

        with st.spinner("Training model (this can take a minute)..."):
            model, X_train, X_test, y_test, params = _train(
                labeled, feature_cols, label_col, retune
            )

        st.session_state["results_ready"] = True
        st.session_state["scored"] = scored
        st.session_state["missing_group_count"] = missing_group_count
        st.session_state["too_few_peers_count"] = too_few_peers_count
        st.session_state["model"] = model
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["identifier_col"] = identifier_col

    if st.session_state.get("results_ready"):
        scored = st.session_state["scored"]
        missing_group_count = st.session_state["missing_group_count"]
        too_few_peers_count = st.session_state["too_few_peers_count"]
        model = st.session_state["model"]
        X_train = st.session_state["X_train"]
        X_test = st.session_state["X_test"]
        id_col = st.session_state["identifier_col"]

        proba = model.predict_proba(X_test)[:, 1]
        display = scored.loc[X_test.index, [id_col]].copy()
        display["confidence"] = proba

        st.success(
            f"Scored {len(scored):,} rows. "
            f"{too_few_peers_count:,} excluded for too few peers, "
            f"{missing_group_count:,} excluded for missing peer-group data "
            f"(blank values in one of the columns used to define the peer group)."
        )

        threshold = st.slider(
            "Flagging sensitivity (lower catches more, with more false alarms)",
            min_value=0.01, max_value=0.99, value=0.30, step=0.01,
        )
        display["flagged"] = display["confidence"] > threshold
        flagged_count = int(display["flagged"].sum())
        st.caption(f"{flagged_count:,} of {len(display):,} test rows flagged at this sensitivity.")

        results = display[display["flagged"]].sort_values("confidence", ascending=False)
        st.subheader("Flagged rows")
        selected = st.dataframe(
            results,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            hide_index=True,
        )

        if selected and selected.selection.rows:
            row_pos = results.index[selected.selection.rows[0]]
            st.subheader(f"Why was this flagged? ({display.loc[row_pos, id_col]})")
            explainer = build_explainer(model, X_train)
            row_X = X_test.loc[[row_pos]]
            pairs = explain_row(explainer, row_X, top_n=8)
            exp_df = pd.DataFrame(pairs, columns=["feature", "impact"]).sort_values("impact")

            fig, ax = plt.subplots(figsize=(7, 0.45 * len(exp_df) + 1), dpi=150)
            colors = ["#2a78d6" if v < 0 else "#eb6834" for v in exp_df["impact"]]
            ax.barh(exp_df["feature"], exp_df["impact"], color=colors)
            ax.axvline(0, color="#898781", linewidth=1)
            ax.set_xlabel("SHAP value (impact on flag decision)")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            st.caption(
                "Orange bars pushed the model toward flagging this row; "
                "blue bars pushed toward normal. Longer bar means bigger impact."
            )
