import streamlit as st
import joblib
import numpy as np

# =========================
# LOAD MODELS
# =========================
kmeans = joblib.load("models/kmeans.pkl")
risk_model = joblib.load("models/risk_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Customer Segmentation & Risk Analysis", layout="wide")

st.title("💳 Customer Segmentation & Default Risk Analysis")
st.write("Analyze customer loan behavior and assess default risk.")

# =========================
# CLUSTER DEFINITIONS
# =========================
cluster_descriptions = {
    0: {
        "title": "Stable Borrower",
        "description": "Customers with healthy payment behavior and low credit exposure.",
        "recommendation": "Offer premium loan products and loyalty rewards."
    },
    1: {
        "title": "Moderate Risk Customer",
        "description": "Customers with moderate debt ratio and occasional late payments.",
        "recommendation": "Offer controlled credit increase with monitoring."
    },
    2: {
        "title": "High Risk / High Utilization",
        "description": "Customers with high debt burden and higher probability of default.",
        "recommendation": "Apply strict credit control and risk monitoring."
    }
}

# =========================
# SHOW ALL CLUSTER INFO
# =========================
st.header("📊 Customer Segmentation Overview")

for cid, info in cluster_descriptions.items():
    with st.expander(f"Cluster {cid} - {info['title']}"):
        st.write("**Description:**", info["description"])
        st.write("**Business Recommendation:**", info["recommendation"])

st.divider()

# =========================
# INPUT FORM
# =========================
st.header("📝 Input Customer Financial Data")

col1, col2 = st.columns(2)

with col1:
    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=8000000.0,
        help="Total monthly income earned by the customer."
    )

    monthly_payment = st.number_input(
        "Monthly Loan Payment",
        min_value=0.0,
        value=2000000.0,
        help="Total monthly loan installment paid by the customer."
    )

    debt_ratio = st.slider(
        "Debt to Income Ratio",
        0.0, 1.0, 0.3,
        help="Proportion of total debt compared to income."
    )

with col2:
    credit_util = st.slider(
        "Credit Utilization Rate",
        0.0, 1.0, 0.4,
        help="Percentage of credit card limit used."
    )

    prev_defaults = st.selectbox(
        "Previous Loan Defaults",
        [0, 1],
        help="1 = Customer has defaulted before, 0 = No default history."
    )

    payment_history = st.slider(
        "Payment History Score",
        0, 100, 80,
        help="Score representing consistency of past loan payments."
    )

# =========================
# ANALYZE BUTTON
# =========================
if st.button("🔍 Analyze Customer"):

    # Feature Engineering (same as training)
    payment_ratio = monthly_payment / monthly_income if monthly_income != 0 else 0

    sample = np.array([[
        payment_ratio,
        debt_ratio,
        credit_util,
        prev_defaults,
        payment_history
    ]])

    # Clustering (scaled)
    sample_scaled = scaler.transform(sample)
    cluster = kmeans.predict(sample_scaled)[0]

    # Risk Prediction (not scaled)
    risk_pred = risk_model.predict(sample)[0]
    risk_prob = risk_model.predict_proba(sample)[0][1]

    st.divider()
    st.header("📈 Analysis Result")

    # =========================
    # CLUSTER RESULT
    # =========================
    cluster_info = cluster_descriptions[cluster]

    st.subheader(f"Customer Segment: {cluster_info['title']}")
    st.write(cluster_info["description"])
    st.info(f"Business Recommendation: {cluster_info['recommendation']}")

    # =========================
    # RISK RESULT
    # =========================
    st.subheader("Default Risk Assessment")

    if risk_pred == 1:
        st.error("⚠ High Default Risk")
    else:
        st.success("✅ Low Default Risk")

    st.metric("Default Probability", f"{risk_prob:.2%}")
