import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def compute_feature_importance(
    model, feature_names: list, top_n: int = 20
) -> pd.DataFrame:
    """
    Extract feature importances from a fitted model.

    Supports:
    - RandomForest / DecisionTree  → .feature_importances_
    - LogisticRegression           → |coefficients| (single class or multi)
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        raise ValueError("Model does not expose feature importances or coefficients.")

    df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False).reset_index(drop=True)

    return df.head(top_n)


def get_churn_risk_level(probability: float) -> str:
    """Map a churn probability to a human-readable risk label."""
    if probability < 0.25:
        return "🟢 Low"
    elif probability < 0.50:
        return "🟡 Medium"
    elif probability < 0.75:
        return "🟠 High"
    else:
        return "🔴 Critical"


def get_risk_color(probability: float) -> str:
    """Return a hex color for the risk level (used in Streamlit tables)."""
    if probability < 0.25:
        return "#28a745"
    elif probability < 0.50:
        return "#ffc107"
    elif probability < 0.75:
        return "#fd7e14"
    else:
        return "#dc3545"
