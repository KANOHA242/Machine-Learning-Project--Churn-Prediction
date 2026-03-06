import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="centered"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e8eaf0;
}

h1, h2, h3 { font-family: 'Space Mono', monospace; }
.stApp { background: #0d0f14; }

section[data-testid="stSidebar"] {
    background: #13161f;
    border-right: 1px solid #1e2233;
}

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00e5ff;
    letter-spacing: -1px;
    margin-bottom: 0;
}

.subtitle {
    color: #5a6080;
    font-size: 0.9rem;
    margin-top: 0.2rem;
    margin-bottom: 2rem;
    font-family: 'Space Mono', monospace;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #00e5ff;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-bottom: 1px solid #1e2233;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    color: #8890b0 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px;
}

div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] input {
    background: #13161f !important;
    border: 1px solid #1e2233 !important;
    color: #e8eaf0 !important;
    border-radius: 6px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #00e5ff, #0077ff);
    color: #0d0f14;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 1px;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2.5rem;
    width: 100%;
    transition: all 0.2s ease;
    margin-top: 1rem;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px #00e5ff33;
}

.result-card {
    border-radius: 12px;
    padding: 2rem;
    margin-top: 1.5rem;
    text-align: center;
    border: 1px solid;
}

.result-churn    { background: #1a0f0f; border-color: #ff4757; }
.result-no-churn { background: #0a1a12; border-color: #00e676; }

.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.result-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.5rem;
    font-weight: 700;
    line-height: 1;
}

.result-churn .result-label { color: #ff4757; }
.result-churn .result-value { color: #ff4757; }
.result-no-churn .result-label { color: #00e676; }
.result-no-churn .result-value { color: #00e676; }

.result-desc { font-size: 0.85rem; color: #5a6080; margin-top: 0.75rem; }

.proba-bar-bg {
    background: #1e2233;
    border-radius: 4px;
    height: 6px;
    margin-top: 1.2rem;
    overflow: hidden;
}

.proba-bar-fill { height: 100%; border-radius: 4px; }

.model-badge {
    display: inline-block;
    background: #13161f;
    border: 1px solid #1e2233;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #5a6080;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("src/best_random_forest_model.pkl")
    except Exception as e:
        st.error(f"Erreur chargement modèle : {e}")
        return None

model = load_model()

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📡 ChurnRadar</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">// Customer Churn Prediction System</p>', unsafe_allow_html=True)
if model:
    st.markdown('<span class="model-badge">⬤ Random Forest · Model v1.0</span>', unsafe_allow_html=True)

# ─── Form ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Customer Profile</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    senior_citizen   = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner          = st.selectbox("Partner", ["Yes", "No"])
    dependents       = st.selectbox("Dependents", ["Yes", "No"])
    tenure           = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    multiple_lines   = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security  = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

with col2:
    online_backup     = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support      = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    contract          = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method    = st.selectbox("Payment Method", [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)

# ─── Predict ───────────────────────────────────────────────────────────────────
if st.button("ANALYSER LE RISQUE"):
    if model is None:
        st.error("Modèle non chargé.")
    else:
        # Reconstruction exacte des colonnes get_dummies(drop_first=True)
        # La catégorie de référence (drop_first) est la première par ordre alphabétique
        input_data = pd.DataFrame([{
            # Numériques (inchangés)
            "SeniorCitizen"                         : 1 if senior_citizen == "Yes" else 0,
            "tenure"                                : tenure,
            "MonthlyCharges"                        : monthly_charges,

            # Partner → ref = "No"
            "Partner_Yes"                           : 1 if partner == "Yes" else 0,

            # Dependents → ref = "No"
            "Dependents_Yes"                        : 1 if dependents == "Yes" else 0,

            # MultipleLines → ref = "No"
            "MultipleLines_No phone service"        : 1 if multiple_lines == "No phone service" else 0,
            "MultipleLines_Yes"                     : 1 if multiple_lines == "Yes" else 0,

            # InternetService → ref = "DSL"
            "InternetService_Fiber optic"           : 1 if internet_service == "Fiber optic" else 0,
            "InternetService_No"                    : 1 if internet_service == "No" else 0,

            # OnlineSecurity → ref = "No"
            "OnlineSecurity_No internet service"    : 1 if online_security == "No internet service" else 0,
            "OnlineSecurity_Yes"                    : 1 if online_security == "Yes" else 0,

            # OnlineBackup → ref = "No"
            "OnlineBackup_No internet service"      : 1 if online_backup == "No internet service" else 0,
            "OnlineBackup_Yes"                      : 1 if online_backup == "Yes" else 0,

            # DeviceProtection → ref = "No"
            "DeviceProtection_No internet service"  : 1 if device_protection == "No internet service" else 0,
            "DeviceProtection_Yes"                  : 1 if device_protection == "Yes" else 0,

            # TechSupport → ref = "No"
            "TechSupport_No internet service"       : 1 if tech_support == "No internet service" else 0,
            "TechSupport_Yes"                       : 1 if tech_support == "Yes" else 0,

            # Contract → ref = "Month-to-month"
            "Contract_One year"                     : 1 if contract == "One year" else 0,
            "Contract_Two year"                     : 1 if contract == "Two year" else 0,

            # PaperlessBilling → ref = "No"
            "PaperlessBilling_Yes"                  : 1 if paperless_billing == "Yes" else 0,

            # PaymentMethod → ref = "Bank transfer (automatic)"
            "PaymentMethod_Credit card (automatic)" : 1 if payment_method == "Credit card (automatic)" else 0,
            "PaymentMethod_Electronic check"        : 1 if payment_method == "Electronic check" else 0,
            "PaymentMethod_Mailed check"            : 1 if payment_method == "Mailed check" else 0,
        }])

        prediction  = model.predict(input_data)[0]
        proba       = model.predict_proba(input_data)[0]
        churn_proba = proba[1] * 100

        if prediction == 1:
            st.markdown(f"""
            <div class="result-card result-churn">
                <p class="result-label">Résultat de l'analyse</p>
                <p class="result-value">CHURN</p>
                <p class="result-desc">Ce client présente un risque élevé de désabonnement.</p>
                <div class="proba-bar-bg">
                    <div class="proba-bar-fill" style="width:{churn_proba:.0f}%; background:#ff4757;"></div>
                </div>
                <p style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#ff4757;margin-top:0.5rem;">
                    Probabilité de churn : {churn_proba:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-no-churn">
                <p class="result-label">Résultat de l'analyse</p>
                <p class="result-value">FIDÈLE</p>
                <p class="result-desc">Ce client est susceptible de rester abonné.</p>
                <div class="proba-bar-bg">
                    <div class="proba-bar-fill" style="width:{100-churn_proba:.0f}%; background:#00e676;"></div>
                </div>
                <p style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#00e676;margin-top:0.5rem;">
                    Probabilité de rétention : {100-churn_proba:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)