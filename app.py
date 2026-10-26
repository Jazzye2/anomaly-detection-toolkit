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

RAW_BASE = "https://raw.githubusercontent.com/Jazzye2/anomaly-detection-toolkit/master/examples/data"

st.set_page_config(page_title="Anomaly Detection Toolkit", page_icon="🔎", layout="wide")
st.title("Anomaly Detection Toolkit")
st.caption(
    "Peer-group anomaly scoring with an explainable classifier on top. "
    "Upload a dataset, tell it what to compare, and get a reviewable list of flags."
)

st.info(
    "**Built toward real Medicare fraud detection.** This method, compare a "
    "provider to its true peers rather than to everyone, directly mirrors how "
    "real CMS/OIG program integrity analytics work. A concrete example: "
    "**upcoding**, billing a high-complexity visit code (CPT 99215) when a "
    "lower one was actually warranted, is a well-documented real fraud pattern. "
    "Compare each provider's rate of high-complexity billing against same-"
    "specialty, same-region peers, and a provider billing far outside that norm "
    "is worth a second look. Same scoring method and same SHAP explainability "
    "already built here, pointed at the right data. "
    "[Full roadmap to a production system](https://github.com/Jazzye2/anomaly-detection-toolkit#what-would-actually-be-needed-for-real-medicare-fraud-detection)."
)

with st.expander("How does this work?"):
    st.markdown(
        """
Each row gets compared only to its **peers**, rows that share the same category
(for example, the same county and the same staffing type). A row is flagged if
its value is unusually far from what's typical for its specific peer group, not
just unusual compared to the dataset as a whole.

A model then learns the pattern behind those flags, so it can explain *why* a
row was flagged (the bar chart you'll see after selecting a row) and let you
adjust how aggressive the flagging is with a single slider, instead of digging
through statistics yourself.

The **confidence** number is the model's own estimate of how likely a row is
to be a genuine anomaly, not a certainty. Treat flagged rows as "worth a
second look," not as a verdict.
        """
    )

with st.expander("New here? Try one of these example datasets first"):
    st.markdown(
        f"""
**Hospital staffing** — [download CSV]({RAW_BASE}/hospital_staffing.csv)

- Identifier column: `Facility Name`
- Value to screen for anomalies: `Productive Hours per Adjusted Patient Day`
- Peer group columns: `County Name`, `Type of Control`, `Hours Type`
- Extra context columns: those three plus `Year`, `Productive Hours`

**Animal control incidents** — [download CSV]({RAW_BASE}/animal_control_cleaned.csv)

- Identifier column: `FILE NUMBER`
- Value to screen for anomalies: `RESPONSE_MINUTES`
- Peer group columns: `SPECIES`, `REQUEST TYPE`, `MUNICIPALITY`
- Extra context columns: those three plus `SEX`, `AGE`, `DISPOSITION`
- Already cleaned: the raw dataset only gives dispatch/arrival timestamps,
  not a ready response-time column, and about half its timestamps turned
  out to be placeholders rather than real data. This file has that worked
  out already (response time computed, placeholder timestamps and
  missing categories removed), so you can go straight to running it.
        """
    )

with st.sidebar:
    st.header("1. Data")
    uploaded = st.file_uploader("CSV file", type="csv")

    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            st.error(
                "Couldn't read that file as a CSV. Double check it's actually a CSV "
                "(not an Excel file renamed to .csv), and that it isn't empty or corrupted."
            )

    if df is not None and df.empty:
        st.error("That file has no rows in it.")
        df = None

    if df is not None:
        columns = list(df.columns)
        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]

        if not numeric_cols:
            st.error(
                "This file has no numeric columns, so there's nothing to screen for "
                "anomalies. This tool needs at least one column of numbers (like a "
                "rate, a duration, or a count)."
            )
            st.stop()

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


if df is None:
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
        st.session_state["results_file_id"] = uploaded.file_id
        st.session_state["scored"] = scored
        st.session_state["missing_group_count"] = missing_group_count
        st.session_state["too_few_peers_count"] = too_few_peers_count
        st.session_state["model"] = model
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["identifier_col"] = identifier_col

    results_match_current_file = (
        st.session_state.get("results_ready")
        and st.session_state.get("results_file_id") == uploaded.file_id
    )
    if st.session_state.get("results_ready") and not results_match_current_file:
        st.info(
            "You've uploaded a different file than the one these settings were last "
            "run against. Click **Run analysis** to see results for this file."
        )

    if results_match_current_file:
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

        scoreable_count = len(scored) - too_few_peers_count - missing_group_count
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows scored", f"{len(scored):,}")
        m2.metric("Successfully evaluated", f"{scoreable_count:,}")
        m3.metric("Excluded, too few peers", f"{too_few_peers_count:,}")
        m4.metric("Excluded, missing data", f"{missing_group_count:,}")
        st.caption(
            "\"Too few peers\" means the peer group had fewer rows than the minimum size. "
            "\"Missing data\" means one of the peer-group columns was blank for that row."
        )

        threshold = st.slider(
            "Flagging sensitivity",
            min_value=0.01, max_value=0.99, value=0.30, step=0.01,
            help=(
                "How confident the model needs to be before it flags a row. "
                "Move it left to catch more real cases, at the cost of more false "
                "alarms. Move it right to see only the cases it's most sure about, "
                "fewer false alarms, but some real cases might slip through."
            ),
        )
        display["flagged"] = display["confidence"] > threshold
        flagged_count = int(display["flagged"].sum())

        if threshold < 0.2:
            tone = "**Aggressive.** Catching more potential cases, expect more false alarms mixed in."
        elif threshold > 0.5:
            tone = "**Conservative.** Only the cases the model is most confident about, some real cases may be missed."
        else:
            tone = "**Balanced.** A middle ground between catching more and staying precise."

        st.caption(
            f"**{flagged_count:,} of {len(display):,}** rows flagged at this setting. {tone} "
            f"There's no universally correct setting, it depends on whether a wasted review "
            f"or a missed case is the costlier mistake for your situation."
        )

        results = display[display["flagged"]].sort_values("confidence", ascending=False)
        st.subheader("Flagged rows")
        st.download_button(
            "Download flagged rows as CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="flagged_rows.csv",
            mime="text/csv",
        )
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

st.divider()
st.caption(
    "This is a demonstration of a peer-group anomaly detection methodology, not a "
    "production fraud-detection system, and it isn't connected to any real Medicare "
    "or CMS data. "
    "[Source code & methodology](https://github.com/Jazzye2/anomaly-detection-toolkit)"
)
