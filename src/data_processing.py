import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# Columns that are binary yes/no (will be label-encoded)
BINARY_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

# Columns with more than 2 categories (will be one-hot encoded)
MULTICAT_COLS = ["InternetService", "Contract", "PaymentMethod", "gender"]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV and do basic type fixes."""
    df = pd.read_csv(filepath)

    # Remove leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()

    # Drop customerID – not a feature
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges may have spaces instead of values
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    return df


def preprocess(df: pd.DataFrame):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X : np.ndarray  – feature matrix (scaled)
    y : np.ndarray  – target labels (0/1)
    scaler : fitted StandardScaler
    feature_names : list[str]
    encoders : dict  – label encoders keyed by column name
    """
    df = df.copy()

    # ── Target ────────────────────────────────────────────────────────────────
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    # Drop rows where target is missing
    df = df.dropna(subset=["Churn"])
    y = df["Churn"].values.astype(int)
    df = df.drop(columns=["Churn"])

    encoders: dict = {}

    # ── Binary columns → label encode ─────────────────────────────────────────
    for col in BINARY_COLS:
        if col in df.columns:
            le = LabelEncoder()
            # Map 'No internet service' / 'No phone service' → 'No' first
            df[col] = df[col].replace(
                {"No internet service": "No", "No phone service": "No"}
            )
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # ── Multi-category columns → one-hot encode ────────────────────────────────
    cols_to_encode = [c for c in MULTICAT_COLS if c in df.columns]
    if cols_to_encode:
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    # ── Numeric columns → scale ────────────────────────────────────────────────
    scaler = StandardScaler()
    numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
    df[numeric_present] = scaler.fit_transform(df[numeric_present])

    feature_names = list(df.columns)
    X = df.values.astype(float)

    return X, y, scaler, feature_names, encoders


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Stratified 80/20 train/test split."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def preprocess_uploaded(df: pd.DataFrame, scaler, feature_names: list, encoders: dict):
    """
    Preprocess a user-uploaded DataFrame using already-fitted transformers.
    Used at inference time (no target column expected / optional).
    """
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Handle target column if present (remove for prediction)
    has_churn = "Churn" in df.columns
    if has_churn:
        df = df.drop(columns=["Churn"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Binary encode
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].replace(
                {"No internet service": "No", "No phone service": "No"}
            )
            if col in encoders:
                le = encoders[col]
                df[col] = df[col].astype(str).map(
                    lambda x, le=le: le.transform([x])[0]
                    if x in le.classes_ else 0
                )
            else:
                df[col] = (df[col].astype(str).str.lower() == "yes").astype(int)

    # One-hot encode
    cols_to_encode = [c for c in MULTICAT_COLS if c in df.columns]
    if cols_to_encode:
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    # Align columns with training features
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Scale numerics
    numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
    df[numeric_present] = scaler.transform(df[numeric_present])

    return df.values.astype(float)
