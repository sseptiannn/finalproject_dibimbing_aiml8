import streamlit as st
import joblib
import numpy as np
import json
import ollama

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Segmentation & Risk Analysis",
    layout="wide"
)

# =========================
# LOAD MODELS
# =========================
kmeans = joblib.load("models/kmeans.pkl")
risk_model = joblib.load("models/risk_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# =========================
# LLM FUNCTION
# =========================
def generate_llm_recommendation(result_json):

    json_string = json.dumps(result_json, indent=2)

    prompt = f"""
    You are a senior credit risk analyst.

    Here is the customer analysis result:

    {json_string}

    Please provide:
    1. Risk interpretation
    2. Business recommendation
    3. Suggested loan strategy
    4. Risk mitigation advice

    Keep it professional, clear, and concise.
    """

    response = ollama.chat(
        model="llama3",  # ganti ke "phi3" kalau mau lebih ringan
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


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
# SHOW CLUSTER INFO
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
    monthly_income = st.number_input("Monthly Income", min_value=0.0, value=8000000.0)
    monthly_payment = st.number_input("Monthly Loan Payment", min_value=0.0, value=2000000.0)
    debt_ratio = st.slider("Debt to Income Ratio", 0.0, 1.0, 0.3)

with col2:
    credit_util = st.slider("Credit Utilization Rate", 0.0, 1.0, 0.4)
    prev_defaults = st.selectbox("Previous Loan Defaults", [0, 1])
    payment_history = st.slider("Payment History Score", 0, 100, 80)

# =========================
# ANALYZE BUTTON
# =========================
if st.button("🔍 Analyze Customer"):

    payment_ratio = monthly_payment / monthly_income if monthly_income != 0 else 0

    processed_features = {
        "payment_ratio": payment_ratio,
        "debt_income_ratio": debt_ratio,
        "credit_utilization": credit_util,
        "PreviousLoanDefaults": prev_defaults,
        "PaymentHistory": payment_history
    }

    sample = np.array([list(processed_features.values())])

    sample_scaled = scaler.transform(sample)
    cluster = int(kmeans.predict(sample_scaled)[0])

    risk_pred = int(risk_model.predict(sample)[0])
    risk_prob = float(risk_model.predict_proba(sample)[0][1])

    cluster_info = cluster_descriptions[cluster]

    result_json = {
        "input_features": processed_features,
        "cluster_result": {
            "cluster_id": cluster,
            "cluster_name": cluster_info["title"]
        },
        "risk_analysis": {
            "prediction": risk_pred,
            "default_probability": round(risk_prob, 4)
        }
    }

    # SAVE TO SESSION
    st.session_state.result_json = result_json
    st.session_state.cluster_info = cluster_info
    st.session_state.risk_pred = risk_pred
    st.session_state.risk_prob = risk_prob

# =========================
# DISPLAY RESULTS (IF AVAILABLE)
# =========================
if "result_json" in st.session_state:

    st.divider()
    st.header("📈 Analysis Result")

    cluster_info = st.session_state.cluster_info
    risk_pred = st.session_state.risk_pred
    risk_prob = st.session_state.risk_prob

    st.subheader(f"Customer Segment: {cluster_info['title']}")
    st.write(cluster_info["description"])
    st.info(f"Business Recommendation: {cluster_info['recommendation']}")

    st.subheader("Default Risk Assessment")

    if risk_pred == 1:
        st.error("⚠ High Default Risk")
    else:
        st.success("✅ Low Default Risk")

    st.metric("Default Probability", f"{risk_prob:.2%}")

    st.subheader("📦 Model Output (JSON)")
    st.json(st.session_state.result_json)

    st.download_button(
        label="⬇ Download Result as JSON",
        data=json.dumps(st.session_state.result_json, indent=4),
        file_name="customer_analysis.json",
        mime="application/json"
    )

    # =========================
    # AI INTERPRETATION
    # =========================
    st.divider()
    st.header("🤖 AI Risk Interpretation")

    if st.button("Generate AI Recommendation"):

        with st.spinner("Generating AI insight..."):
            llm_output = generate_llm_recommendation(
                st.session_state.result_json
            )

        st.markdown(llm_output)
