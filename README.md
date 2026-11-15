# Customer Churn Prediction System

Predicts whether a telecom customer will churn (cancel their subscription) using
**Logistic Regression** and **Random Forest**, with an interactive **Streamlit** dashboard.

## Problem
Binary classification: `1` = churn, `0` = stays.
Dataset: IBM Telco Customer Churn (7,043 customers, 20 features). About 26.5% of customers churn,
so the classes are imbalanced.

## Approach
- **EDA** (`notebooks/churn_analysis.ipynb`): churn vs contract, tenure, monthly charges, payment method, internet service.
- **Preprocessing** (`src/preprocessing.py`): fix `TotalCharges` (text -> number), drop `customerID`, median imputation,
  `StandardScaler` for numbers, One-Hot Encoding for categories, stratified train/test split.
  Everything lives in a scikit-learn `Pipeline` + `ColumnTransformer`, so it is fitted on training data only (no data leakage).
- **Feature selection** (`src/feature_selection.py`): correlation, mutual information and Random Forest importance, validated with cross-validation.
- **Models** (`src/train.py`): Logistic Regression (baseline) and Random Forest, tuned with `GridSearchCV` / `RandomizedSearchCV`
  (5-fold CV, ROC-AUC). `class_weight="balanced"` handles imbalance. The final model is chosen by CV score on training data only.
- **Evaluation** (`src/evaluate.py`): accuracy, precision, recall, F1, confusion matrix, ROC curve, ROC-AUC, threshold trade-off.
- **App** (`app.py`): loads the saved pipeline (no retraining) and shows overview, analysis, single-customer prediction, retention suggestions, model info.

## Results (test set)
Run `python -m src.train` and copy your own numbers (also saved in `models/comparison.csv`).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ___ | ___ | ___ | ___ | ___ |
| Random Forest | ___ | ___ | ___ | ___ | ___ |

Selected model: ___ (reason: ___)

## Project structure
```
customer-churn-prediction/
├── data/Telco-Customer-Churn.csv
├── notebooks/churn_analysis.ipynb
├── src/  data_loader.py  preprocessing.py  feature_selection.py  train.py  evaluate.py  predict.py
├── models/   (created by training: churn_pipeline.pkl, metrics.json, feature_importance.csv, comparison.csv)
├── reports/  (created by training: roc_curve.png, confusion_matrix.png)
├── tests/test_pipeline.py
├── app.py
├── requirements.txt
└── README.md
```

## Run locally
```bash
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python -m src.train              # trains, evaluates, saves the pipeline
python -m src.predict            # sample prediction
python -m tests.test_pipeline    # tests
streamlit run app.py             # dashboard at http://localhost:8501
```

## Limitations
Single dataset, no time dimension, no cost data for retention offers, the 0.5 threshold is not cost-optimised,
correlation is not causation, and predictions support decisions but do not guarantee behaviour.

## Future work
Cost-based threshold, SHAP explanations, drift monitoring, scheduled retraining, API deployment, A/B testing retention offers.
# Customer_churn
