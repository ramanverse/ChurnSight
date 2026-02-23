# ChurnSight — Customer Churn Prediction System

> An AI-driven analytics dashboard that predicts which telecom customers are likely to churn, surfaces the key drivers behind churn, and lets you drill into individual customer risk profiles — all in a lightweight Streamlit app.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Architecture & Pipeline](#architecture--pipeline)
5. [Models](#models)
6. [Dataset](#dataset)
7. [Installation](#installation)
8. [Usage](#usage)
   - [Running the App](#running-the-app)
   - [Training via CLI](#training-via-cli)
9. [App Pages](#app-pages)
10. [Configuration & Customisation](#configuration--customisation)
11. [Dependencies](#dependencies)

---

## Project Overview

ChurnSight is a customer analytics system built on historical data from the IBM Telco Customer Churn dataset to:

- Identify customers at risk of leaving.
- Quantify each customer's churn probability (0–100 %).
- Explain *why* a customer might churn through feature importance analysis.
- Allow business teams to filter, explore, and download at-risk customer lists.

---

## Features

| Feature | Details |
|---|---|
| 📊 Dataset Overview | Summary KPIs, churn distribution pie, monthly-charge histogram |
| 🤖 Model Training | One-click training of Logistic Regression and Decision Tree classifiers |
| 📈 Evaluation | Side-by-side ROC curves, confusion matrices, and a full performance table (Accuracy, Precision, Recall, F1, ROC-AUC) |
| 🔮 Batch Prediction | Run predictions on the built-in dataset or any uploaded CSV |
| 🎯 Risk Levels | Automatic bucketing into 🟢 Low / 🟡 Medium / 🟠 High / 🔴 Critical |
| 🔍 Feature Importance | Interactive bar chart for any trained model, including a full sortable table |
| 💾 Export | Download predictions as CSV and charts as PNG |
| ⚡ Persistent Artifacts | Trained models are saved to `models/churn_models.pkl` and reloaded automatically |

---

## Project Structure

```
AIML Project/
├── app.py                          # Streamlit UI — all 5 pages
├── requirements.txt                # Pinned Python dependencies
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # IBM Telco dataset
├── models/
│   └── churn_models.pkl            # Saved model artifacts (generated after training)
└── src/
    ├── __init__.py
    ├── data_preprocessing.py       # Load, clean, encode, scale, split
    ├── feature_engineering.py      # Feature importance extraction & risk labelling
    ├── model.py                    # Train, evaluate, save, load models
    └── train.py                    # Standalone CLI training script
```

---

## Architecture & Pipeline

```
Raw CSV
   │
   ▼
data_preprocessing.py
   │  ├─ Drop customerID
   │  ├─ Fix TotalCharges (coerce → fill median)
   │  ├─ Label-encode binary columns (Yes/No)
   │  ├─ One-hot encode multi-category columns
   │  └─ StandardScaler on numeric columns
   │
   ▼
model.py  ──  train_models()
   │  ├─ Logistic Regression  (max_iter=1000)
   │  └─ Decision Tree        (max_depth=8)
   │
   ▼
evaluate_all_models()  →  Accuracy / Precision / Recall / F1 / ROC-AUC
   │
   ▼
save_artifacts()  →  models/churn_models.pkl  (joblib)
   │
   ▼
Streamlit app  →  Predict, explain, export
```

At **inference time**, uploaded CSVs are preprocessed with the *already-fitted* scaler and label encoders (stored inside the pkl), so the feature space always matches training.

---

## Models

| Model | Key Hyperparameters |
|---|---|
| **Logistic Regression** | `max_iter=1000`, `random_state=42` |
| **Decision Tree** | `max_depth=8`, `random_state=42` |

The best model (highest ROC-AUC) is automatically highlighted in the UI and pre-selected in the Predict and Feature Importance pages.

---

## Dataset

**IBM Telco Customer Churn**
[Kaggle link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property | Value |
|---|---|
| Rows | 7,043 customers |
| Columns | 21 (20 features + 1 target) |
| Target | `Churn` — Yes / No |

### Key Features Used

| Column | Type | Description |
|---|---|---|
| `tenure` | Numeric | Months the customer has been with the company |
| `MonthlyCharges` | Numeric | Current monthly bill ($) |
| `TotalCharges` | Numeric | Cumulative spend ($) |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `InternetService` | Categorical | DSL / Fiber optic / No |
| `PaymentMethod` | Categorical | Electronic check / Mail check / Bank transfer / Credit card |
| `OnlineSecurity` | Binary | Yes / No |
| `TechSupport` | Binary | Yes / No |
| `Partner` | Binary | Yes / No |
| `Dependents` | Binary | Yes / No |

---

## Installation

### Prerequisites

- Python **3.9 – 3.12**
- `pip`

### Steps

```bash
# 1. Clone / download the repository
cd "AIML Project"

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip3 install -r requirements.txt
```

---

## Usage

### Running the App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** by default.

### Training via CLI

If you prefer to train from the terminal without launching the UI:

```bash
python src/train.py
```

This runs the full pipeline — load → preprocess → train → evaluate → save — and prints a formatted results table to stdout. The artifact is saved to `models/churn_models.pkl`.

---

## App Pages

### 1. Overview
High-level KPIs from the bundled dataset: total customers, churned count, retention count, and churn rate. Includes a churn distribution donut chart and a monthly-charges histogram overlaid by churn status.

### 2. Train & Evaluate
- Click **Train All Models** to run the full training pipeline.
- View a performance summary table for all models.
- Explore ROC curves (one line per model) and confusion matrices side by side.
- The best model by ROC-AUC is highlighted automatically.

### 3. Predict
- Select any trained model from the dropdown (best model pre-selected).
- Use the **bundled Telco dataset** or upload your own CSV.
- See aggregate KPIs, a risk-distribution donut, and a probability histogram.
- Filter the customer risk table by risk tier and download results as CSV.

### 4. Feature Importance
- Choose a model to inspect.
- Adjust the slider to show the top N features (5–30).
- View an interactive horizontal bar chart and a full sortable table.

### 5. About
Full data dictionary, model outputs reference, and tech-stack summary.

---

## Configuration & Customisation

| What | Where | How |
|---|---|---|
| Add / remove models | `src/model.py` → `train_models()` | Add a new key–value pair; evaluation and UI pick it up automatically |
| Change train/test split | `src/data_preprocessing.py` → `split_data()` | Adjust `test_size` (default `0.2`) |
| Change risk thresholds | `src/feature_engineering.py` → `get_churn_risk_level()` | Edit the probability cut-offs |
| Change dataset path | `app.py` and `src/train.py` | Update the `DATASET` constant / `load_data()` call |
| Retrain after code changes | UI or CLI | Delete `models/churn_models.pkl` and re-train |

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | 1.35.0 | Web UI |
| `pandas` | 2.2.3 | Data manipulation |
| `numpy` | 1.26.4 | Numerical operations |
| `scikit-learn` | 1.4.2 | ML models, preprocessing, metrics |
| `plotly` | 5.22.0 | Interactive charts |
| `joblib` | 1.4.2 | Model serialisation |
| `imbalanced-learn` | 0.12.3 | (Available for future SMOTE oversampling) |
| `matplotlib` | 3.9.0 | (Available for static plots) |
| `seaborn` | 0.13.2 | (Available for statistical plots) |
| `kaleido` | 0.2.1 | PNG export of Plotly charts |
