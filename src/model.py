import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve
)
import warnings
warnings.filterwarnings("ignore")

MODEL_NAMES = ["Logistic Regression", "Decision Tree"]
MODEL_FILE = "churn_models.pkl"


def train_models(X_train, y_train) -> dict:
    """Train three classifiers and return them in a dict keyed by model name."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


def evaluate_model(model, X_test, y_test) -> dict:
    """Return a dict of classification metrics for one model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "fpr": fpr,
        "tpr": tpr,
    }


def evaluate_all_models(models: dict, X_test, y_test) -> dict:
    """Run evaluate_model for every model in the dict."""
    return {name: evaluate_model(m, X_test, y_test) for name, m in models.items()}


def save_artifacts(models: dict, scaler, feature_names: list, encoders: dict, path: str = "models"):
    """Persist all artifacts to disk."""
    os.makedirs(path, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "scaler": scaler,
            "feature_names": feature_names,
            "encoders": encoders,
        },
        os.path.join(path, MODEL_FILE),
    )


def load_artifacts(path: str = "models") -> dict:
    """Load artifacts from disk. Returns dict with models, scaler, feature_names, encoders."""
    return joblib.load(os.path.join(path, MODEL_FILE))


def get_best_model(models: dict, metrics: dict) -> tuple:
    """Return (name, model) of the model with the highest ROC-AUC."""
    best_name = max(metrics, key=lambda n: metrics[n]["roc_auc"])
    return best_name, models[best_name]
