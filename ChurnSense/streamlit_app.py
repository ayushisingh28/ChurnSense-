import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ChurnSense" / "model.sav"
DATA_PATH = BASE_DIR / "ChurnSense" / "first_telc.csv"

INPUT_COLUMNS = [
    "SeniorCitizen",
    "MonthlyCharges",
    "TotalCharges",
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure",
]

CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


@st.cache_resource
def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_reference_data(path: Path):
    return pd.read_csv(path)


def build_input_dataframe(values: dict) -> pd.DataFrame:
    return pd.DataFrame([values], columns=INPUT_COLUMNS)


def predict_churn(input_df: pd.DataFrame, reference_df: pd.DataFrame, model):
    combined = pd.concat([reference_df, input_df], ignore_index=True)
    labels = [f"{i} - {i + 11}" for i in range(1, 72, 12)]
    combined["tenure_group"] = pd.cut(
        combined["tenure"].astype(int),
        range(1, 80, 12),
        right=False,
        labels=labels,
    )
    combined.drop(columns=["tenure"], axis=1, inplace=True)

    dummies = pd.get_dummies(
        combined[
            [
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
                "tenure_group",
            ]
        ]
    )

    prediction = model.predict(dummies.tail(1))[0]
    probability = model.predict_proba(dummies.tail(1))[:, 1][0]
    return prediction, probability


def main():
    st.set_page_config(page_title="ChurnSense", page_icon="📉", layout="centered")
    st.title("ChurnSense")
    st.write("Advanced customer churn prediction system.")

    st.sidebar.header("Customer profile")
    senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1], index=0)
    monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=70.0, step=1.0, format="%.2f")
    total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, value=150.0, step=1.0, format="%.2f")
    gender = st.sidebar.selectbox("Gender", CATEGORICAL_OPTIONS["gender"])
    partner = st.sidebar.selectbox("Partner", CATEGORICAL_OPTIONS["Partner"])
    dependents = st.sidebar.selectbox("Dependents", CATEGORICAL_OPTIONS["Dependents"])
    phone_service = st.sidebar.selectbox("Phone Service", CATEGORICAL_OPTIONS["PhoneService"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", CATEGORICAL_OPTIONS["MultipleLines"])
    internet_service = st.sidebar.selectbox("Internet Service", CATEGORICAL_OPTIONS["InternetService"])
    online_security = st.sidebar.selectbox("Online Security", CATEGORICAL_OPTIONS["OnlineSecurity"])
    online_backup = st.sidebar.selectbox("Online Backup", CATEGORICAL_OPTIONS["OnlineBackup"])
    device_protection = st.sidebar.selectbox("Device Protection", CATEGORICAL_OPTIONS["DeviceProtection"])
    tech_support = st.sidebar.selectbox("Tech Support", CATEGORICAL_OPTIONS["TechSupport"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", CATEGORICAL_OPTIONS["StreamingTV"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", CATEGORICAL_OPTIONS["StreamingMovies"])
    contract = st.sidebar.selectbox("Contract", CATEGORICAL_OPTIONS["Contract"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", CATEGORICAL_OPTIONS["PaperlessBilling"])
    payment_method = st.sidebar.selectbox("Payment Method", CATEGORICAL_OPTIONS["PaymentMethod"])
    tenure = st.sidebar.number_input("Tenure (months)", min_value=0, value=12, step=1)

    if st.button("Predict churn"):
        model = load_model(MODEL_PATH)
        reference_df = load_reference_data(DATA_PATH)

        input_values = {
            "SeniorCitizen": senior_citizen,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "gender": gender,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "tenure": tenure,
        }

        input_df = build_input_dataframe(input_values)
        prediction, probability = predict_churn(input_df, reference_df, model)

        if prediction == 1:
            st.error("This customer is likely to churn.")
        else:
            st.success("This customer is likely to continue.")

        st.write(f"**Confidence:** {probability * 100:.2f}%")

    st.markdown(
        "---\n" 
        "#### Notes\n"
        "- The model uses a reference dataset to align categorical encodings.\n"
        "- Enter realistic values for `MonthlyCharges`, `TotalCharges`, and `tenure`."
    )


if __name__ == "__main__":
    main()
