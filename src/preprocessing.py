import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ADDON_COLUMNS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                 "TechSupport", "StreamingTV", "StreamingMovies"]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "NumServices"]

CATEGORICAL_FEATURES = ["gender", "SeniorCitizen", "Partner", "Dependents",
                        "PhoneService", "MultipleLines", "InternetService"] + ADDON_COLUMNS + [
                        "Contract", "PaperlessBilling", "PaymentMethod"]

# Columns a raw customer record must contain (NumServices is computed, not typed in)
RAW_FEATURES = [c for c in NUMERIC_FEATURES if c != "NumServices"] + CATEGORICAL_FEATURES
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_features(df):
    """Fix types and add NumServices. Used for training AND for new customers."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # blanks -> NaN
    if pd.api.types.is_numeric_dtype(df["SeniorCitizen"]):
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["NumServices"] = (df[ADDON_COLUMNS] == "Yes").sum(axis=1)
    return df


def clean_data(df):
    """Clean the full training dataset (has customerID and Churn)."""
    df = df.copy()
    print("Duplicate customerIDs:", df["customerID"].duplicated().sum())
    df = df.drop_duplicates(subset="customerID")
    df = df.drop(columns=["customerID"])           # an ID has no predictive meaning
    df = prepare_features(df)
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df


def split_data(df, test_size=0.2, random_state=42):
    """Split into train/test. stratify keeps the churn % the same in both."""
    X = df.drop(columns=["Churn"])[ALL_FEATURES]
    y = df["Churn"]
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def build_preprocessor(numeric_features=None, categorical_features=None):
    """ColumnTransformer: impute+scale numbers, one-hot encode categories."""
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES

    numeric_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_steps, numeric_features),
        ("cat", categorical_steps, categorical_features),
    ])


if __name__ == "__main__":
    from src.data_loader import load_data
    df = clean_data(load_data("data/Telco-Customer-Churn.csv"))
    print(df.shape)
    print("Missing values:\n", df.isnull().sum()[lambda s: s > 0])
    X_train, X_test, y_train, y_test = split_data(df)
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Churn rate train/test:", y_train.mean().round(3), y_test.mean().round(3))
    pre = build_preprocessor()
    X_train_t = pre.fit_transform(X_train)     # fit on TRAIN only
    X_test_t = pre.transform(X_test)           # only transform on test
    print("Transformed train shape:", X_train_t.shape)
