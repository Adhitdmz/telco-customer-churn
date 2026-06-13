import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("churn_model.pkl")

st.set_page_config(
    page_title="Telco Customer Churn",
    page_icon="📞",
    layout="wide"
)

# Sidebar
st.sidebar.title("📊 Model Information")

st.sidebar.success("Model : Random Forest")

st.sidebar.metric(
    label="Accuracy",
    value="79.56%"
)

st.sidebar.markdown("### Top Features")

st.sidebar.write("""
1. TotalCharges
2. MonthlyCharges
3. tenure
4. Contract
5. PaymentMethod
""")

st.sidebar.markdown("---")

st.sidebar.write("""
### Tentang Project

Prediksi Customer Churn pada perusahaan Telco menggunakan Machine Learning Random Forest.
""")

# Main Page
st.title("📞 Telco Customer Churn Prediction")

st.write("Masukkan data pelanggan untuk memprediksi churn.")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    Partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    Dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (bulan)",
        0,
        100,
        12
    )

    PhoneService = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    MultipleLines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    InternetService = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    OnlineSecurity = st.selectbox(
        "Online Security",
        ["Yes", "No"]
    )

    OnlineBackup = st.selectbox(
        "Online Backup",
        ["Yes", "No"]
    )

with col2:

    DeviceProtection = st.selectbox(
        "Device Protection",
        ["Yes", "No"]
    )

    TechSupport = st.selectbox(
        "Tech Support",
        ["Yes", "No"]
    )

    StreamingTV = st.selectbox(
        "Streaming TV",
        ["Yes", "No"]
    )

    StreamingMovies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No"]
    )

    Contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    PaperlessBilling = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    MonthlyCharges = st.number_input(
        "Monthly Charges",
        0.0,
        200.0,
        70.0
    )

    TotalCharges = st.number_input(
        "Total Charges",
        0.0,
        10000.0,
        1000.0
    )

if st.button("🔮 Prediksi Customer"):

    data = pd.DataFrame({
        'gender':[gender],
        'SeniorCitizen':[SeniorCitizen],
        'Partner':[Partner],
        'Dependents':[Dependents],
        'tenure':[tenure],
        'PhoneService':[PhoneService],
        'MultipleLines':[MultipleLines],
        'InternetService':[InternetService],
        'OnlineSecurity':[OnlineSecurity],
        'OnlineBackup':[OnlineBackup],
        'DeviceProtection':[DeviceProtection],
        'TechSupport':[TechSupport],
        'StreamingTV':[StreamingTV],
        'StreamingMovies':[StreamingMovies],
        'Contract':[Contract],
        'PaperlessBilling':[PaperlessBilling],
        'PaymentMethod':[PaymentMethod],
        'MonthlyCharges':[MonthlyCharges],
        'TotalCharges':[TotalCharges]
    })

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ Customer Diprediksi Churn")
    else:
        st.success("✅ Customer Diprediksi Tidak Churn")

st.markdown("---")

st.subheader("📖 Tentang Proyek")

st.write("""
Dataset: Telco Customer Churn

Model: Random Forest Classifier

Accuracy: 79.56%

Top Feature Importance:
- TotalCharges
- MonthlyCharges
- tenure
- Contract
- PaymentMethod
""")
