import streamlit as st
import joblib
import numpy as np
import json
import os

# OPTIONAL: ollama (safe import)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

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
model_path = "models"

# ---- Load clustering models
clustering_models = {}
for f in os.listdir(model_path):
    if "clustering_model" in f and f.endswith(".pkl"):
        name = f.replace(".pkl", "")
        clustering_models[name] = joblib.load(os.path.join(model_path, f))

# ---- Load scaler
scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))

# ---- Load risk models
risk_models = {}
for f in os.listdir(model_path):
    if f.startswith("risk_model_") and f.endswith(".pkl"):
        name = f.replace(".pkl", "")
        risk_models[name] = joblib.load(os.path.join(model_path, f))

# =========================
# CLUSTER DESCRIPTIONS (PER MODEL)
# =========================
cluster_descriptions = {

    # ===== KMEANS (2 CLUSTERS) =====
    "clustering_model_kmeans": {
        0: {
            "title": "Stable Borrower",
            "description": "Customers with healthy repayment behavior and low credit exposure.",
            "recommendation": "Offer premium loan products and loyalty rewards."
        },
        1: {
            "title": "Higher Risk Borrower",
            "description": "Customers with higher payment pressure or past payment issues.",
            "recommendation": "Apply stricter credit control and closer monitoring."
        }
    },

    # ===== GMM (3 CLUSTERS) =====
    "clustering_model_gmm": {
        0: {
            "title": "Low Risk Segment",
            "description": "Customers with strong financial stability and consistent payment behavior.",
            "recommendation": "Provide cross-selling and loyalty programs."
        },
        1: {
            "title": "Moderate Risk Segment",
            "description": "Customers with moderate debt burden and some risk indicators.",
            "recommendation": "Offer controlled credit expansion with monitoring."
        },
        2: {
            "title": "High Risk Segment",
            "description": "Customers showing high utilization and higher default tendency.",
            "recommendation": "Limit exposure and implement proactive risk mitigation."
        }
    }
}

# =========================
# LLM FUNCTION
# =========================
def generate_llm_recommendation(result_json):
    if not OLLAMA_AVAILABLE:
        return "⚠️ Ollama is not installed in this environment."

    json_string = json.dumps(result_json, indent=2)

    prompt = f"""
You are a senior credit risk analyst.

Here is the customer analysis result:

{json_string}

Provide:
1. Risk interpretation
2. Business recommendation
3. Suggested loan strategy
4. Risk mitigation advice

Keep it professional and concise.
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# =========================
# SESSION STATE
# =========================
if "result_json" not in st.session_state:
    st.session_state.result_json = None
    st.session_state.llm_output = None

# =========================
# HEADER
# =========================
st.title("AI-Powered Customer Segmentation & Default Risk Intelligence System")
st.write(
    "Analyze customer loan behavior and assess default risk. "
    "Input financial data to receive segmentation insights and risk predictions."
)

# =========================
# INPUT FORM
# =========================
st.header("📝 Input Customer Financial Data")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        monthly_income = st.number_input("Monthly Income", min_value=0.0, value=8000000.0)
        monthly_payment = st.number_input("Monthly Loan Payment", min_value=0.0, value=2000000.0)
        debt_ratio = st.slider("Debt to Income Ratio", 0.0, 1.0, 0.3)

    with col2:
        credit_util = st.slider("Credit Utilization Rate", 0.0, 1.0, 0.4)
        prev_defaults = st.selectbox("Previous Loan Defaults", [0, 1])
        payment_history = st.slider("Payment History Score", 0, 100, 80)

    submitted = st.form_submit_button("🔍 Analyze Customer")

# =========================
# ANALYSIS
# =========================
if submitted:
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

    result_json = {
        "input_features": processed_features,
        "clustering_results": {},
        "risk_results": {}
    }

    # ===== CLUSTERING =====
    for cname, cmodel in clustering_models.items():
        cluster_id = int(cmodel.predict(sample_scaled)[0])

        # 🔥 IMPORTANT FIX (per-model description)
        model_desc = cluster_descriptions.get(cname, {})
        desc = model_desc.get(cluster_id, {})

        result_json["clustering_results"][cname] = {
            "cluster_id": cluster_id,
            "cluster_name": desc.get("title", ""),
            "description": desc.get("description", ""),
            "recommendation": desc.get("recommendation", "")
        }

    # ===== RISK MODELS =====
    for rname, rmodel in risk_models.items():
        risk_pred = int(rmodel.predict(sample)[0])
        risk_prob = float(rmodel.predict_proba(sample)[0][1])

        result_json["risk_results"][rname] = {
            "prediction": risk_pred,
            "default_probability": round(risk_prob, 4)
        }

    st.session_state.result_json = result_json
    st.session_state.llm_output = None

# =========================
# DISPLAY RESULTS
# =========================
if st.session_state.result_json is not None:
    st.divider()
    st.header("📊 Customer Analysis Results")

    # ===== CLUSTERING =====
    st.subheader("Segmentation per Model")

    for cname, info in st.session_state.result_json["clustering_results"].items():
        pretty_name = cname.replace("clustering_model_", "").upper()

        st.markdown(f"### 🏷 Model: {pretty_name}")

        if "High" in info["cluster_name"]:
            st.error(f"**Segment:** {info['cluster_name']}")
        elif "Moderate" in info["cluster_name"]:
            st.warning(f"**Segment:** {info['cluster_name']}")
        else:
            st.success(f"**Segment:** {info['cluster_name']}")

        st.write(info["description"])
        st.info(f"Business Recommendation: {info['recommendation']}")

    st.divider()

    # ===== RISK =====
    st.subheader("Default Risk per Model")

    for rname, info in st.session_state.result_json["risk_results"].items():
        label = "⚠ High Default Risk" if info["prediction"] == 1 else "✅ Low Default Risk"
        st.metric(rname, label)
        st.markdown(f"**Probability:** {info['default_probability']:.2%}")

    st.divider()

    # ===== JSON =====
    with st.expander("📦 Full JSON Output"):
        st.json(st.session_state.result_json)
        st.download_button(
            label="⬇ Download Result as JSON",
            data=json.dumps(st.session_state.result_json, indent=4),
            file_name="customer_analysis.json",
            mime="application/json"
        )

    # ===== LLM =====
    st.divider()
    st.header("🤖 AI Interpretation")

    if st.button("Generate AI Recommendation"):
        with st.spinner("Generating AI insight..."):
            st.session_state.llm_output = generate_llm_recommendation(
                st.session_state.result_json
            )

    if st.session_state.llm_output is not None:
        st.markdown(st.session_state.llm_output)