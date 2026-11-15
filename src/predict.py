import joblib
import pandas as pd

from src.preprocessing import ALL_FEATURES, RAW_FEATURES, prepare_features

MODEL_PATH = "models/churn_pipeline.pkl"
THRESHOLD = 0.5

SAMPLE_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "No",
    "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 286.5,
}


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def validate_input(df):
    """Raise ValueError with a clear message for bad input."""
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for col in ["tenure", "MonthlyCharges"]:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isnull().any():
            raise ValueError(f"'{col}' must be a number and cannot be empty.")
        if (values < 0).any():
            raise ValueError(f"'{col}' cannot be negative.")


def predict_customers(model, df):
    """Return a copy of df with ChurnProbability and PredictedChurn columns."""
    validate_input(df)
    X = prepare_features(df)[ALL_FEATURES]
    proba = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["ChurnProbability"] = proba
    out["PredictedChurn"] = (proba >= THRESHOLD).astype(int)
    return out


def risk_category(prob):
    if prob < 0.30:
        return "Low"
    if prob < 0.60:
        return "Medium"
    return "High"


if __name__ == "__main__":
    model = load_model()
    result = predict_customers(model, pd.DataFrame([SAMPLE_CUSTOMER]))
    p = result.loc[0, "ChurnProbability"]
    print(f"Churn probability: {p:.1%} | Risk: {risk_category(p)} | "
          f"Predicted churn: {bool(result.loc[0, 'PredictedChurn'])}")
