import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import load_data, preprocess, split_data, preprocess_uploaded
from src.model import train_models, evaluate_all_models, save_artifacts, load_artifacts, get_best_model, MODEL_FILE
from src.feature_engineering import compute_feature_importance, get_churn_risk_level

st.set_page_config(
    page_title="ChurnSight",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper functions ---

def _artifacts_exist():
    return os.path.exists(os.path.join("models", MODEL_FILE))

@st.cache_data(show_spinner=False)
def _load_default_df():
    path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def _ensure_models():
    if "models" not in st.session_state:
        if _artifacts_exist():
            arts = load_artifacts()
            st.session_state.models = arts["models"]
            st.session_state.scaler = arts["scaler"]
            st.session_state.feature_names = arts["feature_names"]
            st.session_state.encoders = arts["encoders"]
            st.session_state.metrics = None
        else:
            return False
    return True

# Plotly chart theme (light)
PLOTLY_THEME = dict(
    paper_bgcolor="white",
    plot_bgcolor="#f9fafb",
    font=dict(color="#2d3748", family="sans-serif"),
    xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#cbd5e0"),
    yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#cbd5e0"),
)

# --- Sidebar ---

with st.sidebar:
    st.title("ChurnSight")
    st.caption("Customer Churn Prediction")

    page = st.radio(
        "Page",
        ["Overview", "Train & Evaluate", "Predict", "Feature Importance", "About"],
        label_visibility="collapsed",
    )

    st.divider()
    df_bundle = _load_default_df()

    if _artifacts_exist():
        st.success("Models trained")
    else:
        st.warning("Not trained yet")

    if df_bundle is not None:
        st.caption(f"Dataset: {len(df_bundle):,} rows")

# --- Page 1: Overview ---

if page == "Overview":
    st.title("ChurnSight")
    st.caption("Predict which customers are likely to churn using machine learning.")
    st.divider()

    if df_bundle is not None:
        churn_count = int((df_bundle["Churn"] == "Yes").sum())
        stay_count = len(df_bundle) - churn_count
        churn_pct = churn_count / len(df_bundle) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", f"{len(df_bundle):,}")
        c2.metric("Churned", f"{churn_count:,}")
        c3.metric("Retained", f"{stay_count:,}")
        c4.metric("Churn Rate", f"{churn_pct:.1f}%")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Churn Distribution")
            fig = go.Figure(go.Pie(
                labels=["Retained", "Churned"],
                values=[stay_count, churn_count],
                hole=0.55,
                marker=dict(colors=["#4299e1", "#fc8181"]),
            ))
            fig.update_layout(**PLOTLY_THEME, height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Monthly Charges by Churn")
            fig2 = go.Figure()
            for label, color, val in [("Retained", "#4299e1", "No"), ("Churned", "#fc8181", "Yes")]:
                subset = df_bundle[df_bundle["Churn"] == val]["MonthlyCharges"]
                fig2.add_trace(go.Histogram(x=subset, name=label, opacity=0.75, marker_color=color, nbinsx=30))
            fig2.update_layout(**PLOTLY_THEME, barmode="overlay", height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("How it works")
    st.code("""
CSV file -> Clean data -> Encode features -> Scale -> Train models -> Predict churn
    """, language=None)

    st.subheader("Key Features in the Dataset")
    st.dataframe(pd.DataFrame({
        "Feature": ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService",
                    "OnlineSecurity", "TechSupport", "PaymentMethod", "Dependents", "Partner"],
        "Type": ["Numeric", "Numeric", "Numeric", "Categorical", "Categorical",
                 "Binary", "Binary", "Categorical", "Binary", "Binary"],
        "Description": [
            "Months with the company", "Monthly bill ($)", "Total billed ($)",
            "Month-to-month / One year / Two year", "DSL / Fiber optic / No",
            "Has online security", "Has tech support",
            "How the customer pays",
            "Has dependents", "Has a partner",
        ]
    }), use_container_width=True, hide_index=True)

# --- Page 2: Train & Evaluate ---

elif page == "Train & Evaluate":
    st.title("Train & Evaluate Models")

    if df_bundle is None:
        st.error("Dataset not found. Add `WA_Fn-UseC_-Telco-Customer-Churn.csv` to the project folder.")
        st.stop()

    if st.button("Train All Models", type="primary"):
        with st.spinner("Training... this takes about 30 seconds"):
            df = load_data("WA_Fn-UseC_-Telco-Customer-Churn.csv")
            X, y, scaler, feature_names, encoders = preprocess(df)
            X_train, X_test, y_train, y_test = split_data(X, y)
            models = train_models(X_train, y_train)
            metrics = evaluate_all_models(models, X_test, y_test)
            save_artifacts(models, scaler, feature_names, encoders)
            st.session_state.update(dict(
                models=models, scaler=scaler, feature_names=feature_names,
                encoders=encoders, metrics=metrics, X_test=X_test, y_test=y_test,
            ))
        st.success("Done! Models saved.")

    # Load from disk if already trained
    if "metrics" not in st.session_state or st.session_state.metrics is None:
        if _artifacts_exist():
            arts = load_artifacts()
            st.session_state.update(arts)
            df = load_data("WA_Fn-UseC_-Telco-Customer-Churn.csv")
            X, y, _, _, _ = preprocess(df)
            _, X_test, _, y_test = split_data(X, y)
            st.session_state.metrics = evaluate_all_models(st.session_state.models, X_test, y_test)
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
        else:
            st.info("Click Train All Models to begin.")
            st.stop()

    metrics = st.session_state.metrics
    models = st.session_state.models

    st.subheader("Performance Summary")
    st.dataframe(pd.DataFrame([{
        "Model": name,
        "Accuracy": f"{m['accuracy']:.4f}",
        "Precision": f"{m['precision']:.4f}",
        "Recall": f"{m['recall']:.4f}",
        "F1": f"{m['f1']:.4f}",
        "ROC-AUC": f"{m['roc_auc']:.4f}",
    } for name, m in metrics.items()]), use_container_width=True, hide_index=True)

    best_name, _ = get_best_model(models, metrics)
    st.info(f"**Best model:** {best_name} — ROC-AUC = {metrics[best_name]['roc_auc']:.4f}")

    st.subheader("ROC Curves")
    roc_fig = go.Figure()
    for (name, m), color in zip(metrics.items(), ["#4299e1", "#ed8936"]):
        roc_fig.add_trace(go.Scatter(
            x=m["fpr"], y=m["tpr"],
            name=f"{name} (AUC={m['roc_auc']:.3f})",
            mode="lines", line=dict(color=color, width=2),
        ))
    roc_fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="gray", dash="dash"), showlegend=False,
    ))
    roc_fig.update_layout(**PLOTLY_THEME, height=400,
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(roc_fig, use_container_width=True)

    st.subheader("Confusion Matrices")
    for col, (name, m) in zip(st.columns(2), metrics.items()):
        cm = m["confusion_matrix"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=["Pred: No", "Pred: Yes"], y=["Actual: No", "Actual: Yes"],
            text=cm, texttemplate="%{text}",
            colorscale=[[0, "#edf2f7"], [1, "#3182ce"]], showscale=False,
        ))
        fig.update_layout(**PLOTLY_THEME, title=name, height=260, margin=dict(t=40, b=10))
        col.plotly_chart(fig, use_container_width=True)

# --- Page 3: Predict ---

elif page == "Predict":
    st.title("Churn Prediction")

    if not _ensure_models():
        st.warning("No trained models found. Train them first.")
        st.stop()

    models = st.session_state.models
    scaler = st.session_state.scaler
    feature_names = st.session_state.feature_names
    encoders = st.session_state.encoders
    best_name, _ = get_best_model(
        models,
        st.session_state.get("metrics") or {n: {"roc_auc": 0} for n in models}
    )

    model_choice = st.selectbox("Model", list(models.keys()), index=list(models.keys()).index(best_name))
    selected_model = models[model_choice]
    st.divider()

    use_bundled = st.checkbox("Use bundled Telco dataset", value=True)
    if use_bundled and df_bundle is not None:
        upload_df = df_bundle.copy()
        st.info(f"Using bundled dataset — {len(upload_df):,} customers")
    else:
        uploaded = st.file_uploader("Upload your CSV", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV with the same columns as the Telco dataset.")
            st.stop()
        upload_df = pd.read_csv(uploaded)

    cust_ids = upload_df["customerID"].values if "customerID" in upload_df.columns else \
               [f"CUST-{i+1:04d}" for i in range(len(upload_df))]
    has_labels = "Churn" in upload_df.columns

    with st.spinner("Running predictions..."):
        X_new = preprocess_uploaded(upload_df, scaler, feature_names, encoders)
        probs = selected_model.predict_proba(X_new)[:, 1]
        preds = (probs >= 0.5).astype(int)

    n_churn = preds.sum()
    n_stay = len(preds) - n_churn

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", f"{len(preds):,}")
    c2.metric("At Risk", f"{n_churn:,}")
    c3.metric("Retained", f"{n_stay:,}")
    c4.metric("Avg Probability", f"{probs.mean():.1%}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Distribution")
        risk_labels = [get_churn_risk_level(p) for p in probs]
        risk_counts = pd.Series(risk_labels).value_counts()
        color_map = {"Low": "#48bb78", "Medium": "#ecc94b", "High": "#ed8936", "Critical": "#fc8181"}
        labels = [l.split()[-1] for l in risk_counts.index]
        donut = go.Figure(go.Pie(
            labels=labels, values=risk_counts.values, hole=0.55,
            marker=dict(colors=[color_map.get(l, "#4299e1") for l in labels]),
        ))
        donut.update_layout(**PLOTLY_THEME, height=280, margin=dict(t=10, b=10))
        st.plotly_chart(donut, use_container_width=True)
        st.download_button("Download PNG", data=donut.to_image(format="png", scale=2),
                           file_name="risk_distribution.png", mime="image/png")

    with col2:
        st.subheader("Churn Probability Histogram")
        hist = go.Figure(go.Histogram(
            x=probs, nbinsx=30,
            marker=dict(color=probs, colorscale=[[0, "#48bb78"], [0.5, "#ecc94b"], [1, "#fc8181"]]),
        ))
        hist.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)
        hist.update_layout(**PLOTLY_THEME, height=280, margin=dict(t=10, b=10),
                           xaxis_title="Probability", yaxis_title="Count")
        st.plotly_chart(hist, use_container_width=True)
        st.download_button("Download PNG", data=hist.to_image(format="png", scale=2),
                           file_name="churn_histogram.png", mime="image/png")

    st.subheader("Customer Risk Table")
    result_df = pd.DataFrame({
        "Customer ID": cust_ids,
        "Churn Probability": probs,
        "Prediction": ["Yes" if p else "No" for p in preds],
        "Risk Level": risk_labels,
    })
    if has_labels:
        result_df["Actual"] = upload_df["Churn"].values

    result_df = result_df.sort_values("Churn Probability", ascending=False).reset_index(drop=True)
    result_df["Churn Probability"] = result_df["Churn Probability"].map(lambda x: f"{x:.2%}")

    risk_filter = st.multiselect("Filter by risk", ["Critical", "High", "Medium", "Low"],
                                 default=["Critical", "High"])
    if risk_filter:
        show_df = result_df[result_df["Risk Level"].apply(lambda l: any(f in l for f in risk_filter))]
    else:
        show_df = result_df

    st.dataframe(show_df, use_container_width=True, hide_index=True, height=400)
    st.caption(f"Showing {len(show_df):,} of {len(result_df):,} customers")

    st.divider()
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button("Download CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv", mime="text/csv", use_container_width=True)
    with dl2:
        bar = go.Figure(go.Bar(
            x=["Total", "At Risk", "Retained"],
            y=[len(preds), int(n_churn), int(n_stay)],
            marker_color=["#4299e1", "#fc8181", "#48bb78"],
        ))
        bar.update_layout(**PLOTLY_THEME, height=300)
        st.download_button("Download Summary Chart",
            data=bar.to_image(format="png", scale=2),
            file_name="churn_summary.png", mime="image/png", use_container_width=True)

# --- Page 4: Feature Importance ---

elif page == "Feature Importance":
    st.title("Feature Importance")

    if not _ensure_models():
        st.warning("No trained models found. Train them first.")
        st.stop()

    models = st.session_state.models
    feature_names = st.session_state.feature_names

    model_choice = st.selectbox("Model", list(models.keys()), index=1)
    top_n = st.slider("How many features to show", 5, min(30, len(feature_names)), 15)

    importance_df = compute_feature_importance(models[model_choice], feature_names, top_n=top_n)

    fig = go.Figure(go.Bar(
        x=importance_df["Importance"],
        y=importance_df["Feature"],
        orientation="h",
        marker=dict(
            color=importance_df["Importance"],
            colorscale=[[0, "#edf2f7"], [0.5, "#63b3ed"], [1, "#2b6cb0"]],
            showscale=False,
        ),
        text=[f"{v:.4f}" for v in importance_df["Importance"]],
        textposition="outside",
    ))
    fig.update_layout(**PLOTLY_THEME, height=max(350, top_n * 28),
                      xaxis_title="Importance Score", margin=dict(l=20, r=80, t=20, b=20))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    top3 = importance_df.head(3)["Feature"].tolist()
    st.info(f"**Top 3 drivers of churn ({model_choice}):** {top3[0]}  |  {top3[1]}  |  {top3[2]}")

    st.subheader("Full Table")
    st.dataframe(importance_df.style.format({"Importance": "{:.6f}"}),
                 use_container_width=True, hide_index=True)

# --- Page 5: About ---

elif page == "About":
    st.title("About ChurnSight")
    st.info(
        "**Project:** Customer Churn Prediction & Agentic Retention Strategy System  \n"
        "**Dataset:** [IBM Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Columns")
        st.markdown("""
| Field | Type | Description |
|-------|------|-------------|
| `customerID` | String | Unique ID |
| `gender` | Categorical | Male / Female |
| `SeniorCitizen` | Binary | 0 or 1 |
| `Partner` | Binary | Yes / No |
| `Dependents` | Binary | Yes / No |
| `tenure` | Numeric | Months at company |
| `PhoneService` | Binary | Yes / No |
| `MultipleLines` | Categorical | Yes / No / No phone |
| `InternetService` | Categorical | DSL / Fiber / No |
| `OnlineSecurity` | Binary | Yes / No |
| `OnlineBackup` | Binary | Yes / No |
| `DeviceProtection` | Binary | Yes / No |
| `TechSupport` | Binary | Yes / No |
| `StreamingTV` | Binary | Yes / No |
| `StreamingMovies` | Binary | Yes / No |
| `Contract` | Categorical | Month / 1yr / 2yr |
| `PaperlessBilling` | Binary | Yes / No |
| `PaymentMethod` | Categorical | 4 options |
| `MonthlyCharges` | Numeric | Monthly bill ($) |
| `TotalCharges` | Numeric | Total billed ($) |
| `Churn` | Target | Yes / No |
        """)

    with col2:
        st.subheader("What the model outputs")
        st.markdown("""
| Output | Description |
|--------|-------------|
| Churn Prediction | Yes or No |
| Churn Probability | 0.0 to 1.0 |
| Risk Level | Low / Medium / High / Critical |
| Feature Importance | Which features drive churn |
| Model Metrics | Accuracy, F1, ROC-AUC |
        """)

        st.subheader("Tech Stack")
        st.markdown("""
| Component | Library |
|-----------|---------|
| Data | pandas, NumPy |
| Models | scikit-learn |
| Charts | Plotly |
| UI | Streamlit |
| Save/Load | joblib |
        """)