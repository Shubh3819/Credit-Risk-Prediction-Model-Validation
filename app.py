import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import requests
import os
import json
import re

st.set_page_config(
    page_title="Credit Risk Portfolio Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("Credit Risk Portfolio Analytics")
st.caption("Portfolio analytics and interactive dashboard — experimental analytical use only")

st.info(
    "The dashboard reads the five original CSV files directly from the local `data` folder. "
    "No file upload is needed. Optional saved test-set predictions are loaded from `model_artifacts/` "
    "and mapped only after row-index and target-label checks pass."
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

REQUIRED_FILES = {
    "customers": "customers.csv",
    "loan_accounts": "loan_accounts.csv",
    "bureau_credit": "bureau_credit.csv",
    "repayment_history": "repayment_history.csv",
    "collections_events": "collections_events.csv",
}

with st.sidebar:
    st.header("Data source")
    st.caption(f"Reading local files from: `{DATA_DIR}`")
    st.success("Using local CSV files")
    st.divider()
    st.header("Dashboard filters")

missing_files = [
    filename for filename in REQUIRED_FILES.values()
    if not (DATA_DIR / filename).is_file()
]
if missing_files:
    st.error(f"The following files are missing from `{DATA_DIR}`:")
    for filename in missing_files:
        st.write(f"- `{filename}`")
    st.markdown(
        "Create a folder named `data` beside `app.py` and place all five original CSV files in it. "
        "Keep the filenames exactly as listed above."
    )
    st.stop()


@st.cache_data(show_spinner="Loading local CSV files...")
def load_local_csvs(data_dir, file_versions):
    data = {}
    for name, filename in REQUIRED_FILES.items():
        file_path = Path(data_dir) / filename
        frame = pd.read_csv(file_path)
        frame.columns = (
            frame.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")
        )
        data[name] = frame
    return data


# File size and modification time invalidate the cache if a CSV changes.
file_versions = tuple(
    (filename, (DATA_DIR / filename).stat().st_size, (DATA_DIR / filename).stat().st_mtime)
    for filename in REQUIRED_FILES.values()
)
tables = load_local_csvs(str(DATA_DIR), file_versions)

customers = tables["customers"].copy()
loans = tables["loan_accounts"].copy()
bureau = tables["bureau_credit"].copy()
repayments = tables["repayment_history"].copy()
collections = tables["collections_events"].copy()

# Defensive checks for the expected join keys.
required_keys = {
    "customers": (customers, "customer_id"),
    "loan_accounts": (loans, "account_id"),
}
missing_keys = [
    f"{name}.{key}" for name, (frame, key) in required_keys.items()
    if key not in frame.columns
]
if "customer_id" not in customers.columns or "customer_id" not in loans.columns:
    missing_keys.append("customer_id join key in customers/loan_accounts")
if missing_keys:
    st.error("Required column(s) not found: " + ", ".join(sorted(set(missing_keys))))
    st.write("Detected columns:")
    for name, frame in tables.items():
        st.write(f"**{name}:** {', '.join(frame.columns)}")
    st.stop()

# Standardize target when present. The original target is preserved as supplied.
if "target_default" not in loans.columns:
    st.error(
        "`target_default` was not found in loan_accounts.csv. "
        "Check the target column name in your source data before continuing."
    )
    st.write("loan_accounts.csv columns:", list(loans.columns))
    st.stop()

# Aggregate one-to-many tables before joining to loan-level data to avoid row multiplication.
def numeric_aggregates(frame, key, prefix, exclude=None):
    exclude = set(exclude or [])
    if key not in frame.columns:
        return pd.DataFrame(columns=[key])
    num_cols = [
        c for c in frame.select_dtypes(include=np.number).columns
        if c != key and c not in exclude
    ]
    if not num_cols:
        counts = frame.groupby(key).size().rename(f"{prefix}record_count").reset_index()
        return counts
    agg = frame.groupby(key)[num_cols].agg(["mean", "max", "sum", "count"])
    agg.columns = [f"{prefix}{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()
    agg[f"{prefix}record_count"] = frame.groupby(key).size().reindex(agg[key]).to_numpy()
    return agg

bureau_agg = numeric_aggregates(bureau, "customer_id", "bureau_")
repayment_agg = numeric_aggregates(repayments, "account_id", "repayment_")
collection_agg = numeric_aggregates(collections, "account_id", "collection_")

# Include payment-status counts when available.
if "payment_status" in repayments.columns and "account_id" in repayments.columns:
    status_counts = pd.crosstab(repayments["account_id"], repayments["payment_status"])
    status_counts.columns = [f"repayment_status_{str(c).strip().lower().replace(' ', '_')}_count"
                             for c in status_counts.columns]
    status_counts = status_counts.reset_index()
    repayment_agg = repayment_agg.merge(status_counts, on="account_id", how="outer")

portfolio = loans.merge(customers, on="customer_id", how="left", suffixes=("", "_customer"))
if "customer_id" in bureau_agg.columns:
    portfolio = portfolio.merge(bureau_agg, on="customer_id", how="left", validate="many_to_one")
if "account_id" in repayment_agg.columns:
    portfolio = portfolio.merge(repayment_agg, on="account_id", how="left", validate="one_to_one")
if "account_id" in collection_agg.columns:
    portfolio = portfolio.merge(collection_agg, on="account_id", how="left", validate="one_to_one")

# Resolve likely field variants without silently inventing unavailable values.
def first_existing(frame, candidates):
    return next((c for c in candidates if c in frame.columns), None)

income_col = first_existing(portfolio, ["annual_income", "yearly_income", "income"])
employment_col = first_existing(portfolio, ["employment_type", "employment_status", "occupation_sector"])
region_col = first_existing(portfolio, ["region", "state", "city", "city_tier", "geographic_region"])
age_col = first_existing(portfolio, ["age", "customer_age"])
loan_type_col = first_existing(portfolio, ["loan_type", "product_type"])
loan_amount_col = first_existing(portfolio, ["loan_amount", "principal_amount", "sanctioned_amount"])
origination_col = first_existing(portfolio, ["origination_date", "loan_origination_date", "disbursement_date"])
target_col = "target_default"
account_col = "account_id"

portfolio[target_col] = pd.to_numeric(portfolio[target_col], errors="coerce")
if loan_amount_col:
    portfolio[loan_amount_col] = pd.to_numeric(portfolio[loan_amount_col], errors="coerce")
else:
    st.warning("No recognized loan amount column was found. Exposure charts will be unavailable.")
    portfolio["_loan_amount_unavailable"] = np.nan
    loan_amount_col = None
portfolio = portfolio[portfolio[account_col].notna()].copy()

# Optional model evaluation artifacts. Keep them in model_artifacts/ beside app.py:
# model_df_engineered.csv and test_set_predictions.csv.
ARTIFACT_DIR = BASE_DIR / "model_artifacts"
ENGINEERED_PATH = ARTIFACT_DIR / "model_df_engineered.csv"
PREDICTIONS_PATH = ARTIFACT_DIR / "test_set_predictions.csv"

model_predictions = pd.DataFrame()
model_artifact_status = "Model evaluation artifacts not found."
if ENGINEERED_PATH.is_file() and PREDICTIONS_PATH.is_file():
    try:
        engineered = pd.read_csv(ENGINEERED_PATH)
        predictions = pd.read_csv(PREDICTIONS_PATH)

        required_engineered = {"account_id", "target_default"}
        required_predictions = {
            "row_index", "actual_target_default",
            "logistic_probability", "xgboost_probability"
        }
        if not required_engineered.issubset(engineered.columns):
            raise ValueError("Engineered dataset is missing account_id or target_default.")
        if not required_predictions.issubset(predictions.columns):
            raise ValueError("Prediction CSV is missing required prediction columns.")

        predictions["row_index"] = pd.to_numeric(predictions["row_index"], errors="raise").astype(int)
        if predictions["row_index"].duplicated().any():
            raise ValueError("Prediction CSV contains duplicate row_index values.")
        if not predictions["row_index"].between(0, len(engineered) - 1).all():
            raise ValueError("Some prediction row_index values are outside the engineered dataset.")

        mapped = engineered.iloc[predictions["row_index"].to_numpy()][
            ["account_id", "customer_id", "target_default"]
            if "customer_id" in engineered.columns
            else ["account_id", "target_default"]
        ].reset_index(drop=True)
        predictions = predictions.reset_index(drop=True)
        if not np.array_equal(
            pd.to_numeric(mapped["target_default"], errors="coerce").to_numpy(),
            pd.to_numeric(predictions["actual_target_default"], errors="coerce").to_numpy()
        ):
            raise ValueError(
                "Prediction target labels do not match engineered rows. "
                "Do not join these predictions to accounts until row order is reconciled."
            )

        model_predictions = pd.concat([mapped, predictions], axis=1)
        model_predictions = model_predictions.drop(columns=["row_index"], errors="ignore")
        model_predictions = model_predictions.drop_duplicates(subset=["account_id"])
        score_cols = [
            c for c in [
                "account_id", "logistic_probability", "logistic_prediction_at_0.5",
                "xgboost_probability", "xgboost_prediction_at_0.5"
            ] if c in model_predictions.columns
        ]
        portfolio = portfolio.merge(
            model_predictions[score_cols], on="account_id", how="left", validate="one_to_one"
        )
        model_artifact_status = (
            f"Loaded {len(model_predictions):,} verified test predictions; "
            f"{model_predictions['account_id'].nunique():,} unique accounts."
        )
    except Exception as exc:
        model_predictions = pd.DataFrame()
        model_artifact_status = f"Could not load model artifacts safely: {exc}"

if origination_col:
    portfolio[origination_col] = pd.to_datetime(portfolio[origination_col], errors="coerce")
if income_col:
    portfolio[income_col] = pd.to_numeric(portfolio[income_col], errors="coerce")
if age_col:
    portfolio[age_col] = pd.to_numeric(portfolio[age_col], errors="coerce")

# Categorical fields used in profiling.
profile_fields = {
    "Income": income_col,
    "Employment type": employment_col,
    "Region": region_col,
    "Age group": age_col,
    "Loan type": loan_type_col,
}
for label, col in profile_fields.items():
    if col and col in portfolio.columns:
        if label == "Income":
            income_values = portfolio[col]
            portfolio["income_band"] = "Unknown"
            valid_income = income_values.notna()
            if valid_income.sum() > 0:
                ranked = income_values.loc[valid_income].rank(method="first")
                quartiles = pd.qcut(ranked, q=min(4, int(valid_income.sum())), duplicates="drop")
                bin_count = len(quartiles.cat.categories)
                labels = ["Lower income", "Lower-middle", "Upper-middle", "Higher income"][:bin_count]
                portfolio.loc[valid_income, "income_band"] = quartiles.cat.rename_categories(labels).astype(str)
            profile_fields[label] = "income_band"
        elif label == "Age group":
            portfolio["age_band"] = pd.cut(
                portfolio[col],
                bins=[0, 29, 39, 49, 59, np.inf],
                labels=["18–29", "30–39", "40–49", "50–59", "60+"],
                include_lowest=True,
            ).astype("object")
            portfolio["age_band"] = portfolio["age_band"].where(portfolio[col].notna(), "Unknown")
            profile_fields[label] = "age_band"
        else:
            portfolio[col] = portfolio[col].fillna("Unknown").astype(str)

portfolio["target_default"] = portfolio["target_default"].where(
    portfolio["target_default"].isin([0, 1])
)
overall_exposure = portfolio[loan_amount_col].clip(lower=0).sum() if loan_amount_col else np.nan
overall_default_rate = portfolio["target_default"].mean()


def grouped_portfolio_summary(frame, by_cols):
    """Summarize accounts, exposure (when available), and observed default rate."""
    aggregations = {
        "accounts": (account_col, "nunique"),
        "default_rate": (target_col, "mean"),
    }
    if loan_amount_col:
        aggregations["exposure"] = (loan_amount_col, "sum")
    else:
        aggregations["exposure"] = (account_col, "count")
    return frame.groupby(by_cols, dropna=False).agg(**aggregations).reset_index()

# Filter controls
st.sidebar.subheader("Filter portfolio")
filtered = portfolio.copy()
filter_cols = []
for label, col in profile_fields.items():
    if col and col in filtered.columns:
        filter_cols.append((label, col))
        options = sorted(filtered[col].dropna().astype(str).unique().tolist())
        selected = st.sidebar.multiselect(label, options, default=options, key=f"filter_{label}")
        filtered = filtered[filtered[col].astype(str).isin(selected)]

if origination_col:
    valid_dates = filtered[origination_col].dropna()
    if not valid_dates.empty:
        min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
        selected_dates = st.sidebar.date_input(
            "Origination date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            start_date, end_date = pd.Timestamp(selected_dates[0]), pd.Timestamp(selected_dates[1])
            filtered = filtered[
                filtered[origination_col].between(start_date, end_date + pd.Timedelta(days=1))
            ]

# Tabs: analytics and dashboard share the same prepared portfolio.
tab_overview, tab_analytics, tab_models, tab_anomalies, tab_accounts, tab_data = st.tabs(
    ["Portfolio Overview", "Portfolio Analytics", "Model Evaluation", "Anomaly Detection", "Account Explorer", "Data Quality"]
)

with tab_overview:
    st.subheader("Portfolio overview")
    total_accounts = filtered[account_col].nunique()
    exposure = filtered[loan_amount_col].clip(lower=0).sum() if loan_amount_col else np.nan
    default_rate = filtered["target_default"].mean()
    avg_loan = filtered[loan_amount_col].mean() if loan_amount_col else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Loan accounts", f"{total_accounts:,}")
    c2.metric("Total exposure", f"{exposure:,.0f}" if pd.notna(exposure) else "Unavailable")
    c3.metric("Observed default rate", f"{default_rate:.1%}" if pd.notna(default_rate) else "Unavailable")
    c4.metric("Average loan amount", f"{avg_loan:,.0f}" if pd.notna(avg_loan) else "Unavailable")

    if loan_type_col:
        by_loan = grouped_portfolio_summary(filtered, loan_type_col)
        left, right = st.columns(2)
        with left:
            if loan_amount_col:
                st.plotly_chart(px.bar(by_loan, x=loan_type_col, y="exposure",
                                       title="Exposure by loan type"), use_container_width=True)
            else:
                st.info("Exposure chart unavailable because no loan amount column was detected.")
        with right:
            st.plotly_chart(px.bar(by_loan, x=loan_type_col, y="default_rate",
                                   title="Observed default rate by loan type",
                                   labels={"default_rate": "Default rate"}), use_container_width=True)
    if origination_col:
        trend = filtered.dropna(subset=[origination_col]).copy()
        if not trend.empty:
            trend["origination_year"] = trend[origination_col].dt.year
            trend = grouped_portfolio_summary(trend, "origination_year")
            if loan_amount_col:
                st.plotly_chart(px.line(trend, x="origination_year", y="exposure", markers=True,
                                        title="Exposure by origination year"),
                                use_container_width=True)
            st.plotly_chart(px.line(trend, x="origination_year", y="default_rate", markers=True,
                                    title="Observed default rate by origination year"),
                            use_container_width=True)

with tab_analytics:
    st.subheader("Portfolio profiling and business insights")
    st.caption("Rates are descriptive and depend on the supplied target definition and observation window.")
    available_profiles = {k: v for k, v in profile_fields.items() if v and v in filtered.columns}
    for label, col in available_profiles.items():
        with st.expander(f"{label} segmentation", expanded=(label == "Income")):
            segment = grouped_portfolio_summary(filtered, col)
            segment["exposure_share_pct"] = 100 * segment["exposure"] / segment["exposure"].sum() if segment["exposure"].sum() else 0
            segment["default_rate_pct"] = 100 * segment["default_rate"]
            st.dataframe(segment.sort_values("exposure", ascending=False), use_container_width=True)
            if loan_amount_col:
                st.plotly_chart(px.bar(segment, x=col, y="exposure_share_pct",
                                       title=f"Exposure share by {label.lower()}",
                                       labels={"exposure_share_pct": "Exposure share (%)"}),
                                use_container_width=True)
            st.plotly_chart(px.bar(segment, x=col, y="default_rate_pct",
                                   title=f"Observed default rate by {label.lower()}",
                                   labels={"default_rate_pct": "Default rate (%)"}),
                            use_container_width=True)

    if income_col and employment_col:
        st.markdown("#### Cross-dimensional analysis: income × employment")
        cross_col = "income_band"
        cross = grouped_portfolio_summary(filtered, [cross_col, employment_col])
        st.dataframe(cross.sort_values("exposure", ascending=False), use_container_width=True)
        heat = cross.pivot(index=employment_col, columns=cross_col, values="default_rate")
        st.plotly_chart(px.imshow(heat, text_auto=".1%", aspect="auto",
                                  title="Observed default rate: income × employment",
                                  labels=dict(color="Default rate")),
                        use_container_width=True)

    st.markdown("#### Automatically generated segment observations")
    insights = []
    min_segment_size = max(50, int(np.ceil(filtered[account_col].nunique() * 0.01)))
    for label, col in available_profiles.items():
        summary = grouped_portfolio_summary(filtered, col)
        summary = summary[summary["accounts"] >= min_segment_size].dropna(subset=["default_rate"])
        if not summary.empty:
            row = summary.sort_values("default_rate", ascending=False).iloc[0]
            insights.append({
                "Dimension": label,
                "Highest observed default segment": str(row[col]),
                "Accounts": int(row["accounts"]),
                "Exposure": float(row["exposure"]),
                "Observed default rate": float(row["default_rate"]),
                "Business implication": "Review segment performance and underwriting/collections practices before changing policy.",
            })
    if insights:
        st.dataframe(pd.DataFrame(insights), use_container_width=True)
    else:
        st.info("Not enough valid segment data to generate segment observations.")

with tab_models:
    st.subheader("Model evaluation and saved test predictions")
    st.caption("Evaluation reflects the saved test split and threshold 0.5 unless stated otherwise.")
    st.info(model_artifact_status)
    if not model_predictions.empty:
        test_view = model_predictions.copy()
        test_view["actual_target_default"] = pd.to_numeric(test_view["actual_target_default"], errors="coerce")
        for col in ["logistic_probability", "xgboost_probability"]:
            if col in test_view.columns:
                test_view[col] = pd.to_numeric(test_view[col], errors="coerce")

        total_test = len(test_view)
        default_rate_test = test_view["actual_target_default"].mean()
        m1, m2, m3 = st.columns(3)
        m1.metric("Test accounts with scores", f"{total_test:,}")
        m2.metric("Test-set observed default rate", f"{default_rate_test:.1%}")
        m3.metric("Scored share of portfolio", f"{total_test / max(portfolio[account_col].nunique(), 1):.1%}")

        metric_rows = []
        for model_name, prob_col in [
            ("Logistic Regression", "logistic_probability"),
            ("XGBoost", "xgboost_probability")
        ]:
            if prob_col not in test_view.columns:
                continue
            y_true = test_view["actual_target_default"]
            y_pred = (test_view[prob_col] >= 0.5).astype(int)
            valid = y_true.notna() & test_view[prob_col].notna()
            y = y_true[valid].astype(int)
            pred_label = y_pred[valid]
            tp = int(((y == 1) & (pred_label == 1)).sum())
            tn = int(((y == 0) & (pred_label == 0)).sum())
            fp = int(((y == 0) & (pred_label == 1)).sum())
            fn = int(((y == 1) & (pred_label == 0)).sum())
            metric_rows.append({
                "Model": model_name,
                "ROC-AUC (reported)": 0.660 if model_name == "Logistic Regression" else 0.655,
                "True negatives": tn, "False positives": fp,
                "False negatives": fn, "True positives": tp,
                "Default recall": tp / max(tp + fn, 1),
                "Default precision": tp / max(tp + fp, 1),
                "Mean predicted probability": test_view.loc[valid, prob_col].mean()
            })
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True)
        st.markdown("#### Saved risk-score distribution")
        plot_df = test_view.melt(
            id_vars=["account_id", "actual_target_default"],
            value_vars=[c for c in ["logistic_probability", "xgboost_probability"] if c in test_view.columns],
            var_name="model", value_name="predicted_probability"
        )
        st.plotly_chart(
            px.histogram(plot_df, x="predicted_probability", color="model", nbins=30,
                         barmode="overlay", opacity=0.65,
                         title="Predicted probability distribution"),
            use_container_width=True
        )
        st.markdown("#### Test accounts and saved predictions")
        st.dataframe(test_view, use_container_width=True, height=420)
        st.download_button(
            "Download mapped test predictions",
            data=test_view.to_csv(index=False).encode("utf-8"),
            file_name="mapped_test_predictions.csv",
            mime="text/csv"
        )
    else:
        st.warning(
            "Place model_df_engineered.csv and test_set_predictions.csv inside "
            "`model_artifacts/` beside app.py. The dashboard will validate row indices "
            "and target labels before attaching any scores."
        )
    st.warning(
        "Model limitation: target observation window is undocumented. Months 1–3 repayment "
        "features may be post-origination, and bureau reporting dates are unavailable. "
        "Treat scores as preliminary exploratory estimates, not production underwriting decisions."
    )

with tab_anomalies:
    st.subheader("Anomaly detection")
    st.caption(
        "Unsupervised screening for unusual loan profiles. An anomaly is not proof of fraud, "
        "default, or policy violation; every flag requires human review."
    )

    # Use loan/customer profile and loan terms only. Exclude target, identifiers,
    # repayment and collections aggregates to avoid using post-origination behavior.
    # Bureau aggregates are excluded here too because their reporting dates are unknown.
    excluded_tokens = ("repayment_", "collection_", "bureau_", "target_default")
    anomaly_candidates = [
        c for c in [
            loan_amount_col, income_col, age_col, "years_employed", "num_dependents",
            "years_at_residence", "interest_rate_pct", "tenure_months"
        ] if c and c in portfolio.columns
    ]
    anomaly_candidates = list(dict.fromkeys(anomaly_candidates))
    anomaly_candidates = [
        c for c in anomaly_candidates
        if c not in {account_col, "customer_id", target_col}
        and not any(token in c.lower() for token in excluded_tokens)
        and pd.api.types.is_numeric_dtype(portfolio[c])
    ]

    if len(anomaly_candidates) < 2:
        st.warning("At least two numeric origination/profile features are required for anomaly screening.")
    else:
        st.write("Features used:", ", ".join(anomaly_candidates))
        contamination = st.slider(
            "Expected share of unusual records to flag", min_value=0.01,
            max_value=0.15, value=0.03, step=0.01,
            help="This is a screening setting, not an estimated fraud/default rate."
        )
        anomaly_cache_dir = BASE_DIR / "model_artifacts"
        anomaly_cache_dir.mkdir(exist_ok=True)
        anomaly_model_path = anomaly_cache_dir / "isolation_forest_origination.joblib"
        anomaly_signature_path = anomaly_cache_dir / "isolation_forest_features.txt"
        feature_signature = "\n".join(anomaly_candidates)

        # Fit on the full portfolio so filters don't change the reference population.
        # Refit when the chosen contamination setting or feature list changes.
        try:
            cache_ok = (
                anomaly_model_path.is_file() and anomaly_signature_path.is_file()
                and anomaly_signature_path.read_text(encoding="utf-8") == feature_signature
            )
            if cache_ok:
                saved = joblib.load(anomaly_model_path)
                cache_ok = saved.get("contamination") == contamination
            if not cache_ok:
                pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    ("detector", IsolationForest(
                        n_estimators=250, contamination=contamination,
                        random_state=42, n_jobs=-1
                    )),
                ])
                pipeline.fit(portfolio[anomaly_candidates])
                joblib.dump({"pipeline": pipeline, "contamination": contamination}, anomaly_model_path)
                anomaly_signature_path.write_text(feature_signature, encoding="utf-8")
                saved = {"pipeline": pipeline, "contamination": contamination}
            detector_pipeline = saved["pipeline"]
            scores = detector_pipeline.decision_function(portfolio[anomaly_candidates])
            flags = detector_pipeline.predict(portfolio[anomaly_candidates]) == -1
            anomaly_view = portfolio[[account_col, "customer_id", target_col]].copy()
            anomaly_view["anomaly_score"] = scores
            anomaly_view["is_anomaly"] = flags
            anomaly_view["review_status"] = np.where(flags, "Review recommended", "No anomaly flag")
            anomaly_view = anomaly_view.merge(
                portfolio[[account_col] + anomaly_candidates], on=account_col,
                how="left", validate="one_to_one"
            )
            visible = anomaly_view[anomaly_view[account_col].isin(filtered[account_col])].copy()
            total_visible = len(visible)
            flagged_visible = int(visible["is_anomaly"].sum())
            k1, k2, k3 = st.columns(3)
            k1.metric("Accounts in current filters", f"{total_visible:,}")
            k2.metric("Flagged for review", f"{flagged_visible:,}")
            k3.metric("Flag rate in filtered view", f"{flagged_visible / max(total_visible, 1):.1%}")

            show_flags_only = st.checkbox("Show flagged accounts only", value=True)
            display_anomalies = visible[visible["is_anomaly"]] if show_flags_only else visible
            display_anomalies = display_anomalies.sort_values("anomaly_score", ascending=True)
            st.dataframe(display_anomalies.head(1000), use_container_width=True, height=400)
            st.download_button(
                "Download anomaly screening results",
                data=visible.sort_values("anomaly_score").to_csv(index=False).encode("utf-8"),
                file_name="anomaly_screening_results.csv", mime="text/csv"
            )

            if flagged_visible:
                st.markdown("#### Review a flagged account")
                flagged_ids = visible.loc[visible["is_anomaly"], account_col].astype(str).tolist()
                chosen_anomaly = st.selectbox("Flagged account ID", flagged_ids)
                row = visible[visible[account_col].astype(str) == chosen_anomaly].iloc[0]
                st.write("Feature values for review")
                st.dataframe(row[anomaly_candidates].rename("Value").to_frame(), use_container_width=True)
                st.info(
                    "Review the source data and supporting context before taking action. "
                    "The anomaly score measures unusualness under this detector; it is not a default probability."
                )
            st.caption(
                "Method: Isolation Forest with median imputation and standardization. "
                "The detector uses numeric loan terms/customer profile only; target labels, repayment, "
                "collections, and bureau aggregates are excluded. Results are exploratory and should be reviewed by a human."
            )
        except Exception as exc:
            st.error(f"Anomaly detection could not run: {exc}")

@st.dialog("Ask the Portfolio", width="large")
def portfolio_chat_dialog():
    st.write(
        "Ask questions about the portfolio's observed default rate, account counts, exposure, "
        "and available customer/loan segments. Python calculates every displayed number from the "
        "currently filtered portfolio; Ollama only provides a short explanation."
    )
    st.caption(
        "This prototype supports descriptive portfolio questions, not underwriting decisions. "
        "Questions about a quarter or time period are rejected unless the dataset contains a suitable, "
        "documented date and target observation window."
    )

    chat_dimensions = {
        "loan type": (loan_type_col, "Loan type"),
        "employment type": (employment_col, "Employment type"),
        "employment": (employment_col, "Employment type"),
        "region": (region_col, "Region"),
        "geography": (region_col, "Region"),
        "age group": ("age_band" if "age_band" in filtered.columns else age_col, "Age group"),
        "age": ("age_band" if "age_band" in filtered.columns else age_col, "Age group"),
        "income group": ("income_band" if "income_band" in filtered.columns else income_col, "Income segment"),
        "income band": ("income_band" if "income_band" in filtered.columns else income_col, "Income segment"),
        "income": ("income_band" if "income_band" in filtered.columns else income_col, "Income segment"),
    }
    supported_dimension_names = [
        name for name, (column, _) in chat_dimensions.items()
        if column and column in filtered.columns
    ]

    def calculate_portfolio_answer(question, frame):
        """Allow-listed descriptive analytics only; never execute LLM-generated code or SQL."""
        q = question.lower().strip()
        if not q:
            return None, "Please enter a question."
        if any(term in q for term in ["this quarter", "last quarter", "quarterly", "this month", "last month", "year to date", "this year"]):
            return None, (
                "I can't reliably answer that time-period question: the dataset's target observation "
                "window is undocumented, and an origination date is not necessarily the default event date. "
                "Try a question about the currently selected portfolio instead."
            )
        if any(term in q for term in ["why did", "predict", "will default", "should we approve", "approve this", "reject this", "fraud"]):
            return None, (
                "That question is outside this chat's validated descriptive-analytics scope. "
                "Use the Model Evaluation or Account Explorer tabs for exploratory model outputs, and have a risk analyst review decisions."
            )

        dim_col, dim_label = None, None
        # Prefer more specific phrases before broad terms such as 'income' or 'age'.
        for name in sorted(chat_dimensions, key=len, reverse=True):
            col, label = chat_dimensions[name]
            if col and col in frame.columns and re.search(r"\b" + re.escape(name) + r"\b", q):
                dim_col, dim_label = col, label
                break

        metric = None
        if any(x in q for x in ["default rate", "default percentage", "defaults highest", "highest default", "riskier segment", "riskiest segment"]):
            metric = "default_rate"
        elif any(x in q for x in ["exposure", "outstanding amount", "loan amount", "portfolio value"]):
            metric = "exposure"
        elif any(x in q for x in ["how many", "number of accounts", "account count", "most accounts", "largest number"]):
            metric = "accounts"
        elif any(x in q for x in ["average income", "mean income"]):
            metric = "average_income"
        elif any(x in q for x in ["average loan", "average loan amount", "mean loan"]):
            metric = "average_loan"
        elif any(x in q for x in ["overall default", "overall portfolio default", "portfolio default rate", "overall rate"]):
            metric = "default_rate"
        elif any(x in q for x in ["highest", "lowest", "top", "compare", "by ", "across", "segment", "segments"]):
            metric = "default_rate"

        if metric is None:
            return None, (
                "I couldn't map that to a supported portfolio calculation. Try asking for the highest "
                "default rate, exposure, or account count by loan type, employment type, region, age group, or income segment."
            )
        if metric in {"default_rate", "exposure", "accounts"} and not dim_col:
            if metric == "default_rate":
                valid = frame[target_col].dropna()
                if valid.empty:
                    return None, "The filtered portfolio has no valid target_default labels to calculate a default rate."
                result = {"scope": "Current dashboard filters", "accounts_with_known_target": int(valid.shape[0]),
                          "observed_default_rate": float(valid.mean()),
                          "definition": "mean of target_default among rows with target labels"}
                return result, None
            if metric == "exposure":
                if not loan_amount_col:
                    return None, "Exposure cannot be calculated because no recognized loan amount column is available."
                result = {"scope": "Current dashboard filters", "accounts": int(frame[account_col].nunique()),
                          "total_exposure": float(frame[loan_amount_col].clip(lower=0).sum()),
                          "definition": f"sum of non-negative {loan_amount_col}"}
                return result, None
            return {"scope": "Current dashboard filters", "accounts": int(frame[account_col].nunique())}, None

        if not dim_col:
            return None, "Please specify a supported segment such as loan type, employment type, region, age group, or income segment."
        if metric == "average_income":
            if not income_col:
                return None, "No recognized income field is available in the loaded data."
            grouped = frame.groupby(dim_col, dropna=False)[income_col].mean().reset_index(name="average_income")
            grouped = grouped.sort_values("average_income", ascending=False, na_position="last")
        elif metric == "average_loan":
            if not loan_amount_col:
                return None, "No recognized loan amount field is available in the loaded data."
            grouped = frame.groupby(dim_col, dropna=False)[loan_amount_col].mean().reset_index(name="average_loan_amount")
            grouped = grouped.sort_values("average_loan_amount", ascending=False, na_position="last")
        else:
            summary = grouped_portfolio_summary(frame, dim_col)
            if metric == "default_rate":
                grouped = summary.dropna(subset=["default_rate"]).sort_values("default_rate", ascending=False)
            elif metric == "exposure":
                grouped = summary.sort_values("exposure", ascending=False)
            else:
                grouped = summary.sort_values("accounts", ascending=False)
        if grouped.empty:
            return None, f"There are no usable records for {dim_label} in the current filters."

        def clean_value(value):
            if pd.isna(value):
                return "Unknown"
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating, float)):
                return float(value)
            return str(value)

        rows = []
        for _, r in grouped.head(10).iterrows():
            item = {dim_label: clean_value(r[dim_col])}
            for col in grouped.columns:
                if col != dim_col:
                    val = r[col]
                    item[col] = None if pd.isna(val) else clean_value(val)
            rows.append(item)
        return {
            "scope": "Current dashboard filters",
            "segment_dimension": dim_label,
            "requested_metric": metric,
            "ranking": "descending",
            "results": rows,
            "definitions": {
                "default_rate": "mean of target_default among non-missing labels in each segment",
                "exposure": "sum of non-negative loan amount values in each segment; not necessarily current outstanding balance",
                "accounts": "unique account_id count",
            },
        }, None

    if "portfolio_chat_history" not in st.session_state:
        st.session_state.portfolio_chat_history = []

    for msg in st.session_state.portfolio_chat_history[-8:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("data"):
                st.dataframe(pd.DataFrame(msg["data"]), use_container_width=True, hide_index=True)

    with st.form(key="portfolio_chat_form", clear_on_submit=True):
        user_question = st.text_input(
            "Your question",
            placeholder="e.g. Which loan type has the highest default rate?",
            key="portfolio_chat_input",
        )
        submitted = st.form_submit_button("Ask the Portfolio")
    if submitted and user_question.strip():
        st.session_state.portfolio_chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        computed, error = calculate_portfolio_answer(user_question, filtered)
        with st.chat_message("assistant"):
            if error:
                st.warning(error)
                final_answer = error
                data_rows = None
            else:
                # Numeric facts are rendered from Python output, not generated by the LLM.
                results = computed.get("results")
                if results:
                    result_df = pd.DataFrame(results)
                    for col in result_df.columns:
                        if "rate" in col.lower() and pd.api.types.is_numeric_dtype(result_df[col]):
                            result_df[col] = result_df[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "Unavailable")
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    data_rows = results
                else:
                    facts = {k: v for k, v in computed.items() if k not in {"definitions"}}
                    st.json(facts)
                    data_rows = None

                # Free-first configuration: use the Ollama model running on this same machine.
                # A Streamlit Community Cloud server cannot reach Ollama on the user's laptop.
                model_name = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
                api_url = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/api/chat"
                headers = {"Content-Type": "application/json"}
                system_prompt = (
                    "You are a concise credit-risk portfolio analyst. The JSON below contains the ONLY "
                    "verified calculation results. Explain what those results mean in 2-4 sentences. "
                    "Do not invent or calculate any numbers, rates, dates, causes, or facts. Do not introduce "
                    "any numeric values. Do not claim causation. State that default rates are observed target-label "
                    "rates, not necessarily forward-looking predictions. Mention current dashboard filters when relevant. "
                    "If the results do not support a conclusion, say so. This is descriptive analysis, not credit advice."
                )
                prompt = "User question: " + user_question + "\nVerified Python calculation results (JSON):\n" + json.dumps(computed, default=str)
                try:
                    response = requests.post(
                        api_url,
                        headers=headers,
                        json={"model": model_name,
                              "messages": [{"role": "system", "content": system_prompt},
                                           {"role": "user", "content": prompt}],
                              "stream": False,
                              "options": {"temperature": 0, "num_ctx": 2048}},
                        timeout=90,
                    )
                    response.raise_for_status()
                    explanation = response.json().get("message", {}).get("content", "").strip()
                    if not explanation:
                        raise ValueError("The model returned an empty explanation.")
                    # Keep numeric claims exclusively in the Python-rendered result table/JSON.
                    # If the model emits any digits despite instructions, replace its prose.
                    if re.search(r"\d", explanation):
                        explanation = (
                            "The table above contains the verified numeric results calculated from the current "
                            "portfolio filters. Interpret the leading segment only for the metric shown; this is "
                            "descriptive analysis, not evidence of causation or a credit decision. Please review "
                            "the denominator and source records before acting."
                        )
                    st.markdown(explanation)
                    final_answer = explanation
                except requests.exceptions.ConnectionError:
                    final_answer = (
                        "The verified calculation table above is still valid, but the local Ollama model is not reachable. "
                        "For local use, start Ollama on this computer and run `ollama run llama3.2:3b`. "
                        "If this dashboard is running on Streamlit Community Cloud, it cannot access Ollama on your laptop. "
                        "A free deployment therefore needs either a separately hosted model endpoint (if you find a suitable free host) "
                        "or to keep the LLM chat local; no paid API is configured by this app."
                    )
                    st.info(final_answer)
                except Exception as exc:
                    final_answer = f"The verified calculation table is available, but Ollama explanation failed: {exc}"
                    st.info(final_answer)
                st.caption(
                    "Human review required: verify the figures and segment definition before acting. "
                    "False positive: the chat labels a segment as highest-risk when it is not. "
                    "False negative: it misses the segment that actually has the highest observed rate. "
                    "Use the displayed Python-calculated table as the source of truth; the LLM explanation is supplementary."
                )
        st.session_state.portfolio_chat_history.append({
            "role": "assistant", "content": final_answer, "data": data_rows
        })

    with st.expander("Supported questions and safeguards"):
        st.markdown(
            "**Try:** Which loan type has the highest default rate?; Compare exposure by region; "
            "Which employment type has the most accounts?; What is the overall observed default rate?; "
            "Which income segment has the highest default rate?\n\n"
            "**Trigger:** a user submits a natural-language question. The app maps it to an allow-listed Python aggregation. "
            "**Unsupported questions:** forecasts, causal explanations, individual approve/reject recommendations, "
            "fraud determinations, and time periods without a documented outcome date are rejected.\n\n"
            "**False positive:** an incorrect segment is presented as the highest-rate segment. "
            "**False negative:** the true highest-rate segment is missed. **Human in the loop:** a risk analyst checks "
            "the result table, denominator, current filters, and source data before any business decision."
        )

    if st.button("Close chat", key="close_portfolio_chat"):
        st.session_state["portfolio_chat_open"] = False
        st.rerun()

# Floating chat launcher. The Streamlit button opens the modal; CSS pins it to the bottom-right.
st.markdown(
    """<style>
    div.st-key-open_portfolio_chat {
        position: fixed; right: 1.35rem; bottom: 1.35rem; z-index: 999999;
        width: auto; max-width: 4.5rem;
    }
    div.st-key-open_portfolio_chat button {
        width: 3.6rem; height: 3.6rem; border-radius: 50%;
        font-size: 1.45rem; box-shadow: 0 4px 18px rgba(0,0,0,.25);
        border: 1px solid rgba(128,128,128,.45);
    }
    </style>""",
    unsafe_allow_html=True,
)
if st.button("💬", key="open_portfolio_chat", help="Open Ask the Portfolio chat"):
    st.session_state["portfolio_chat_open"] = True
if st.session_state.get("portfolio_chat_open", False):
    portfolio_chat_dialog()


with tab_accounts:
    st.subheader("Account explorer")
    st.caption(
        "Scores shown here are saved test-set predictions mapped to accounts using the verified "
        "engineered-dataset row index. They are not newly generated predictions for the entire portfolio."
    )
    search_text = st.text_input("Search account ID", placeholder="Enter an account ID")
    account_view = filtered.copy()
    if search_text:
        account_view = account_view[
            account_view[account_col].astype(str).str.contains(search_text, case=False, na=False)
        ]
    display_cols = [c for c in [
        account_col, "customer_id", loan_type_col, loan_amount_col, target_col,
        income_col, employment_col, region_col, age_col, origination_col
    ] if c and c in account_view.columns]
    st.dataframe(account_view[display_cols].head(500), use_container_width=True)
    if not account_view.empty:
        chosen = st.selectbox("Select account", account_view[account_col].astype(str).unique().tolist())
        record = account_view[account_view[account_col].astype(str) == chosen].iloc[0]
        st.markdown(f"#### Account: `{chosen}`")
        cols = st.columns(3)
        cols[0].metric("Loan amount", f"{record[loan_amount_col]:,.0f}" if loan_amount_col and pd.notna(record[loan_amount_col]) else "Unavailable")
        cols[1].metric("Observed target", str(int(record[target_col])) if pd.notna(record[target_col]) else "Unknown")
        cols[2].metric("Origination date", str(record[origination_col].date()) if origination_col and pd.notna(record[origination_col]) else "Unavailable")
        if "logistic_probability" in record.index and pd.notna(record.get("logistic_probability")):
            score_cols = st.columns(2)
            score_cols[0].metric("Logistic Regression risk score", f"{record['logistic_probability']:.1%}")
            score_cols[1].metric("XGBoost risk score", f"{record['xgboost_probability']:.1%}" if pd.notna(record.get("xgboost_probability")) else "Unavailable")
            st.caption("Scores are model-estimated probabilities, not calibrated guarantees of default. Only test-set accounts have saved scores.")
        else:
            st.info("No saved test-set model score is available for this account.")
        st.write("Account profile")
        st.json({str(c): (None if pd.isna(record[c]) else str(record[c]))
                 for c in display_cols if c != account_col})

with tab_data:
    st.subheader("Source data checks")
    summary_rows = []
    for name, frame in tables.items():
        summary_rows.append({
            "Table": name,
            "Rows": len(frame),
            "Columns": len(frame.columns),
            "Duplicate rows": int(frame.duplicated().sum()),
            "Missing cells": int(frame.isna().sum().sum()),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
    st.markdown("#### Join checks")
    st.write(f"Loan rows: {len(loans):,}")
    st.write(f"Joined portfolio rows: {len(portfolio):,}")
    st.write(f"Unique account IDs after joins: {portfolio[account_col].nunique():,}")
    if len(portfolio) != len(loans):
        st.warning("The joined dataset row count differs from loan_accounts.csv. Review joins before interpreting results.")
    else:
        st.success("The joins preserved the loan-account row count.")
    st.caption("This is a lightweight dashboard-side check, not a replacement for the full Deliverable 1 audit.")
