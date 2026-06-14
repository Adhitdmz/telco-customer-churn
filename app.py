import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("churn_model_v2.pkl")

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
        ["No", "No internet service", "Yes"]
    )

    OnlineBackup = st.selectbox(
        "Online Backup",
        ["No", "No internet service", "Yes"]
    )

with col2:

    DeviceProtection = st.selectbox(
        "Device Protection",
        ["No", "No internet service", "Yes"]
    )

    TechSupport = st.selectbox(
        "Tech Support",
        ["No", "No internet service", "Yes"]
    )

    StreamingTV = st.selectbox(
        "Streaming TV",
        ["No", "No internet service", "Yes"]
    )

    StreamingMovies = st.selectbox(
        "Streaming Movies",
        ["No", "No internet service", "Yes"]
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
   

    gender_map = {"Female": 0, "Male": 1}
    yes_no_map = {"No": 0, "Yes": 1}

    multiple_map = {
        "No": 0,
        "No phone service": 1,
        "Yes": 2
    }

    internet_map = {
        "DSL": 0,
        "Fiber optic": 1,
        "No": 2
    }

    security_map = {
        "No": 0,
        "No internet service": 1,
        "Yes": 2
    }

    contract_map = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }

    payment_map = {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    }

    data = pd.DataFrame({
        'gender': [gender_map[gender]],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [yes_no_map[Partner]],
        'Dependents': [yes_no_map[Dependents]],
        'tenure': [tenure],
        'PhoneService': [yes_no_map[PhoneService]],
        'MultipleLines': [multiple_map[MultipleLines]],
        'InternetService': [internet_map[InternetService]],
        'OnlineSecurity': [security_map[OnlineSecurity]],
        'OnlineBackup': [security_map[OnlineBackup]],
        'DeviceProtection': [security_map[DeviceProtection]],
        'TechSupport': [security_map[TechSupport]],
        'StreamingTV': [security_map[StreamingTV]],
        'StreamingMovies': [security_map[StreamingMovies]],
        'Contract': [contract_map[Contract]],
        'PaperlessBilling': [yes_no_map[PaperlessBilling]],
        'PaymentMethod': [payment_map[PaymentMethod]],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]
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
