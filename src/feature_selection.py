import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.data_loader import load_data
from src.preprocessing import (CATEGORICAL_FEATURES, NUMERIC_FEATURES,
                               build_preprocessor, clean_data, split_data)


def build_importance_table(X_train, y_train):
    pre = build_preprocessor()
    X_t = pre.fit_transform(X_train)                    # TRAIN data only
    names = pre.get_feature_names_out()
    X_t = pd.DataFrame(X_t, columns=names, index=X_train.index)

    corr = X_t.corrwith(y_train)
    mi = mutual_info_classif(X_t, y_train, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                random_state=42, n_jobs=-1).fit(X_t, y_train)

    table = pd.DataFrame({
        "correlation": corr.values,
        "mutual_info": mi,
        "rf_importance": rf.feature_importances_,
    }, index=names)
    for col in ["correlation", "mutual_info", "rf_importance"]:
        key = table[col].abs()
        table[col + "_rank"] = key.rank(ascending=False)
    return table


def cv_auc(drop_cols, X_train, y_train):
    """Cross-validated ROC-AUC of a simple model after dropping some raw columns."""
    num = [c for c in NUMERIC_FEATURES if c not in drop_cols]
    cat = [c for c in CATEGORICAL_FEATURES if c not in drop_cols]
    pipe = Pipeline([
        ("preprocess", build_preprocessor(num, cat)),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    X = X_train.drop(columns=drop_cols)
    return cross_val_score(pipe, X, y_train, cv=5, scoring="roc_auc").mean()


if __name__ == "__main__":
    df = clean_data(load_data("data/Telco-Customer-Churn.csv"))
    X_train, X_test, y_train, y_test = split_data(df)

    table = build_importance_table(X_train, y_train)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(table.sort_values("rf_importance", ascending=False).round(3).head(20))

    # Edit this list AFTER reading the table. Don't just trust a guess.
    DROP = ["gender", "PhoneService"]
    print("\nCV ROC-AUC, all features :", round(cv_auc([], X_train, y_train), 4))
    print(f"CV ROC-AUC, without {DROP}:", round(cv_auc(DROP, X_train, y_train), 4))
