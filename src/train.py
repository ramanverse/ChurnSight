import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_preprocessing import load_data, preprocess, split_data
from src.model import train_models, evaluate_all_models, save_artifacts, get_best_model

DATASET = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def main():
    print("=" * 60)
    print("  Customer Churn Prediction – Training Pipeline")
    print("=" * 60)

    # ── Load & preprocess ──────────────────────────────────────────
    print(f"\n[1/4] Loading dataset: {DATASET}")
    df = load_data(DATASET)
    print(f"      Rows: {len(df):,}  |  Columns: {df.shape[1]}")

    print("\n[2/4] Preprocessing...")
    X, y, scaler, feature_names, encoders = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"      Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
    print(f"      Features: {len(feature_names)}")
    churn_rate = y.mean() * 100
    print(f"      Churn rate: {churn_rate:.1f}%")

    # ── Train ──────────────────────────────────────────────────────
    print("\n[3/4] Training models...")
    models = train_models(X_train, y_train)
    for name in models:
        print(f"      ✓ {name}")

    # ── Evaluate ───────────────────────────────────────────────────
    print("\n[4/4] Evaluation results:")
    print(f"\n  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
    print("  " + "-" * 70)

    metrics = evaluate_all_models(models, X_test, y_test)
    for name, m in metrics.items():
        print(
            f"  {name:<22} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['roc_auc']:>9.4f}"
        )

    best_name, _ = get_best_model(models, metrics)
    print(f"\n  ⭐  Best model: {best_name} (ROC-AUC = {metrics[best_name]['roc_auc']:.4f})")

    # ── Save ───────────────────────────────────────────────────────
    save_artifacts(models, scaler, feature_names, encoders)
    print("\n  ✅  Artifacts saved to models/churn_models.pkl")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
