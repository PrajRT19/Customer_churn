import matplotlib
matplotlib.use("Agg")          # save plots to files without opening a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def evaluate_model(model, X, y, threshold=0.5):
    """Return a dict of metrics for a fitted pipeline."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def print_report(model, X, y, name, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    print(f"\n===== {name} =====")
    print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y, pred))
    print(classification_report(y, pred, target_names=["No churn", "Churn"]))


def plot_confusion_matrix(model, X, y, name, path):
    ConfusionMatrixDisplay.from_estimator(model, X, y, display_labels=["No churn", "Churn"])
    plt.title(f"Confusion matrix: {name}")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_roc_curves(models, X, y, path):
    fig, ax = plt.subplots()
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X, y, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random guess")
    ax.set_title("ROC curves (test set)")
    ax.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def threshold_table(model, X, y, thresholds=(0.3, 0.4, 0.5, 0.6)):
    """Show the precision/recall trade-off at different thresholds (illustration only)."""
    rows = []
    for t in thresholds:
        m = evaluate_model(model, X, y, threshold=t)
        rows.append({"threshold": t, "precision": m["precision"],
                     "recall": m["recall"], "f1": m["f1"]})
    return pd.DataFrame(rows).round(3)


def get_feature_importance(pipeline):
    """Feature importance (RF) or absolute coefficients (logistic regression)."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    values = model.feature_importances_ if hasattr(model, "feature_importances_") else abs(model.coef_[0])
    return pd.DataFrame({"feature": names, "importance": values}).sort_values(
        "importance", ascending=False).reset_index(drop=True)
