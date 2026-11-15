import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data_loader import load_data
from src.evaluate import (evaluate_model, get_feature_importance, plot_confusion_matrix,
                          plot_roc_curves, print_report, threshold_table)
from src.preprocessing import build_preprocessor, clean_data, split_data

DATA_PATH = "data/Telco-Customer-Churn.csv"
MODEL_PATH = "models/churn_pipeline.pkl"
RANDOM_STATE = 42


def make_pipeline(model):
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def main():
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    df = clean_data(load_data(DATA_PATH))
    X_train, X_test, y_train, y_test = split_data(df, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ---- Model 1: Logistic Regression (tune C = inverse regularisation strength) ----
    lr_search = GridSearchCV(
        make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced")),
        param_grid={"model__C": [0.01, 0.1, 1, 10]},
        scoring="roc_auc", cv=cv, n_jobs=-1)
    lr_search.fit(X_train, y_train)

    # ---- Model 2: Random Forest (random search over hyperparameters) ----
    rf_search = RandomizedSearchCV(
        make_pipeline(RandomForestClassifier(class_weight="balanced",
                                             random_state=RANDOM_STATE, n_jobs=-1)),
        param_distributions={
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 5, 8, 12],
            "model__min_samples_leaf": [1, 5, 10, 20],
            "model__min_samples_split": [2, 10, 20],
            "model__max_features": ["sqrt", 0.3],
        },
        n_iter=20, scoring="roc_auc", cv=cv, random_state=RANDOM_STATE, n_jobs=-1)
    rf_search.fit(X_train, y_train)

    models = {"Logistic Regression": lr_search.best_estimator_,
              "Random Forest": rf_search.best_estimator_}
    cv_scores = {"Logistic Regression": lr_search.best_score_,
                 "Random Forest": rf_search.best_score_}
    best_params = {"Logistic Regression": lr_search.best_params_,
                   "Random Forest": rf_search.best_params_}

    # ---- Selection uses CROSS-VALIDATION on train only (not the test set) ----
    selected = max(cv_scores, key=cv_scores.get)

    # ---- Final test-set evaluation (done once) ----
    rows, test_metrics = [], {}
    for name, model in models.items():
        m = evaluate_model(model, X_test, y_test)
        train_auc = evaluate_model(model, X_train, y_train)["roc_auc"]
        test_metrics[name] = m
        rows.append({"model": name, "cv_roc_auc": cv_scores[name],
                     "train_roc_auc": train_auc, **m})
        print_report(model, X_test, y_test, name)
    comparison = pd.DataFrame(rows).set_index("model").round(3)
    print("\n=== COMPARISON (test set) ===")
    print(comparison.T)
    print("\nSelected model (best CV ROC-AUC):", selected)
    print("\nThreshold trade-off for selected model (illustration):")
    print(threshold_table(models[selected], X_test, y_test))

    # ---- Overfitting demo: an unconstrained forest ----
    deep_rf = make_pipeline(RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=1,
        random_state=RANDOM_STATE, n_jobs=-1)).fit(X_train, y_train)
    print("\nUnconstrained RF  train AUC:",
          round(evaluate_model(deep_rf, X_train, y_train)["roc_auc"], 3),
          "| test AUC:", round(evaluate_model(deep_rf, X_test, y_test)["roc_auc"], 3))

    # ---- Plots ----
    plot_roc_curves(models, X_test, y_test, "reports/roc_curve.png")
    plot_confusion_matrix(models[selected], X_test, y_test, selected,
                          "reports/confusion_matrix.png")

    # ---- Save the complete pipeline (preprocessing + model) ----
    final_pipeline = models[selected]
    joblib.dump(final_pipeline, MODEL_PATH)
    get_feature_importance(final_pipeline).to_csv("models/feature_importance.csv", index=False)
    comparison.to_csv("models/comparison.csv")
    with open("models/metrics.json", "w") as f:
        json.dump({"selected_model": selected,
                   "cv_roc_auc": cv_scores,
                   "test_metrics": test_metrics,
                   "best_params": best_params}, f, indent=2, default=str)

    # ---- Verify the saved model reproduces predictions ----
    loaded = joblib.load(MODEL_PATH)
    assert np.array_equal(loaded.predict(X_test), final_pipeline.predict(X_test)), "Mismatch!"
    print("\nSaved + reloaded pipeline gives identical predictions. OK")


if __name__ == "__main__":
    main()
