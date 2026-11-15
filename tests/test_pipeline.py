import pandas as pd

from src.predict import SAMPLE_CUSTOMER, load_model, predict_customers


def test_model_loads():
    assert load_model() is not None


def test_single_prediction_valid():
    out = predict_customers(load_model(), pd.DataFrame([SAMPLE_CUSTOMER]))
    assert 0.0 <= out.loc[0, "ChurnProbability"] <= 1.0
    assert out.loc[0, "PredictedChurn"] in (0, 1)


def _expect_value_error(df):
    try:
        predict_customers(load_model(), df)
    except ValueError:
        return
    raise AssertionError("should have raised ValueError")


def test_missing_column_raises():
    _expect_value_error(pd.DataFrame([SAMPLE_CUSTOMER]).drop(columns=["Contract"]))


def test_negative_tenure_raises():
    _expect_value_error(pd.DataFrame([{**SAMPLE_CUSTOMER, "tenure": -5}]))


def test_empty_monthly_charges_raises():
    _expect_value_error(pd.DataFrame([{**SAMPLE_CUSTOMER, "MonthlyCharges": None}]))


def test_unknown_category_does_not_crash():
    odd = pd.DataFrame([{**SAMPLE_CUSTOMER, "PaymentMethod": "Crypto"}])
    out = predict_customers(load_model(), odd)   # handle_unknown="ignore"
    assert 0.0 <= out.loc[0, "ChurnProbability"] <= 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
