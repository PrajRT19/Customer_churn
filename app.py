import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.data_loader import load_data
from src.preprocessing import ADDON_COLUMNS, clean_data
from src.predict import THRESHOLD, load_model, predict_customers, risk_category

st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")
sns.set_theme(style="whitegrid")

if not Path("models/churn_pipeline.pkl").exists():
    st.error("Model file not found. Run `python -m src.train` first, then restart the app.")
    st.stop()


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_scored_data():
    df = clean_data(load_data("data/Telco-Customer-Churn.csv"))
    return predict_customers(get_model(), df.drop(columns=["Churn"])).join(df["Churn"])


def customer_form():
    """Collect customer details and return them as a dict."""
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
        monthly = st.number_input("Monthly charges", min_value=0.0, max_value=200.0,
                                  value=70.0, step=0.5)
    with c2:
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple = "No phone service" if phone == "No" else st.selectbox("Multiple lines", ["No", "Yes"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
    with c3:
        if internet == "No":
            addons = {col: "No internet service" for col in ADDON_COLUMNS}
            st.info("No internet: add-on services set to 'No internet service'.")
        else:
            addons = {col: st.selectbox(col, ["No", "Yes"]) for col in ADDON_COLUMNS}
        st.caption("TotalCharges is estimated as tenure x monthly charges.")

    return {"gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multiple, "InternetService": internet, **addons,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": tenure * monthly}


def retention_actions(c):
    """Simple rule-based suggestions from the customer's characteristics."""
    actions = []
    if c["Contract"] == "Month-to-month":
        actions.append("Offer a discount for moving to a 1- or 2-year contract.")
    if c["tenure"] < 12:
        actions.append("Run an onboarding follow-up call in the first months.")
    if c["InternetService"] == "Fiber optic" and c["TechSupport"] == "No":
        actions.append("Offer a free tech-support trial.")
    if c["PaymentMethod"] == "Electronic check":
        actions.append("Incentivise automatic payment (small bill credit).")
    if c["MonthlyCharges"] > 85:
        actions.append("Review the plan; suggest a cheaper bundle if usage is low.")
    if not actions:
        actions.append("Keep the standard engagement; no special action needed.")
    return actions


model = get_model()
st.title("📉 Customer Churn Prediction Dashboard")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Churn Analysis", "Predict a Customer", "Recommendations", "Model Info"])

# ---------------- 1. Overview ----------------
with tab1:
    data = get_scored_data()
    total = len(data)
    n_churn = int(data["PredictedChurn"].sum())
    a, b, c, d = st.columns(4)
    a.metric("Total customers", f"{total:,}")
    b.metric("Predicted to churn", f"{n_churn:,}")
    c.metric("Predicted to stay", f"{total - n_churn:,}")
    d.metric("Predicted churn %", f"{n_churn / total:.1%}")
    st.caption("These are model predictions on the dataset the model was built from "
               "(part of it was used for training), so treat them as a demo, not a forecast.")

# ---------------- 2. Analysis ----------------
with tab2:
    data = get_scored_data()
    labelled = data.assign(ChurnLabel=data["Churn"].map({0: "No", 1: "Yes"}))
    st.caption("Charts below use the actual recorded churn from the dataset.")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.countplot(data=labelled, x="ChurnLabel", ax=ax)
        ax.set_title("Churn distribution")
        st.pyplot(fig)
    with col2:
        rate = data.groupby("Contract")["Churn"].mean().mul(100)
        fig, ax = plt.subplots()
        sns.barplot(x=rate.index, y=rate.values, ax=ax)
        ax.set_ylabel("Churn rate (%)")
        ax.set_title("Churn rate by contract")
        st.pyplot(fig)
    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots()
        sns.histplot(data=labelled, x="tenure", hue="ChurnLabel", bins=30, element="step", ax=ax)
        ax.set_title("Tenure by churn")
        st.pyplot(fig)
    with col4:
        fig, ax = plt.subplots()
        sns.boxplot(data=labelled, x="ChurnLabel", y="MonthlyCharges", ax=ax)
        ax.set_title("Monthly charges by churn")
        st.pyplot(fig)

# ---------------- 3. Individual prediction ----------------
with tab3:
    st.subheader("Enter customer details")
    customer = customer_form()
    if st.button("Predict churn", type="primary"):
        try:
            result = predict_customers(model, pd.DataFrame([customer]))
            prob = float(result.loc[0, "ChurnProbability"])
            risk = risk_category(prob)
            st.session_state["last"] = (customer, prob, risk)
            st.metric("Churn probability", f"{prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0))
            verdict = "WILL LIKELY CHURN" if prob >= THRESHOLD else "likely to STAY"
            message = f"Prediction: customer is {verdict}  |  Risk category: **{risk}**"
            {"Low": st.success, "Medium": st.warning, "High": st.error}[risk](message)
        except ValueError as e:
            st.error(f"Invalid input: {e}")

# ---------------- 4. Recommendations ----------------
with tab4:
    if "last" in st.session_state:
        cust, prob, risk = st.session_state["last"]
        st.subheader(f"Last predicted customer: {risk} risk ({prob:.1%})")
        for action in retention_actions(cust):
            st.write("• " + action)
    else:
        st.info("Make a prediction in the previous tab to see tailored suggestions.")
    st.warning("Predictions support business decisions; they do not guarantee how any "
               "individual customer will behave. Combine them with human judgement.")

# ---------------- 5. Model info ----------------
with tab5:
    metrics_path = Path("models/metrics.json")
    if metrics_path.exists():
        info = json.loads(metrics_path.read_text())
        st.subheader(f"Selected model: {info['selected_model']}")
        st.write("Test-set metrics (both models):")
        st.dataframe(pd.DataFrame(info["test_metrics"]).T.round(3))
        st.write("Best hyperparameters found:", info["best_params"][info["selected_model"]])
    imp_path = Path("models/feature_importance.csv")
    if imp_path.exists():
        imp = pd.read_csv(imp_path).head(15)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=imp, y="feature", x="importance", ax=ax)
        ax.set_title("Top 15 important features")
        st.pyplot(fig)
    for img in ["reports/confusion_matrix.png", "reports/roc_curve.png"]:
        if Path(img).exists():
            st.image(img)
