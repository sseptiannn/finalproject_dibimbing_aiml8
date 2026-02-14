import joblib
import numpy as np

# =========================
# LOAD MODELS
# =========================
kmeans = joblib.load("models/kmeans.pkl")
risk_model = joblib.load("models/risk_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# =========================
# SAMPLE DATA (RAW INPUT)
# =========================
# contoh customer baru
MonthlyLoanPayment = 2000000
MonthlyIncome = 8000000
TotalDebtToIncomeRatio = 0.35
CreditCardUtilizationRate = 0.40
PreviousLoanDefaults = 0
PaymentHistory = 85

# =========================
# FEATURE ENGINEERING
# =========================
payment_ratio = MonthlyLoanPayment / MonthlyIncome
debt_income_ratio = TotalDebtToIncomeRatio
credit_utilization = CreditCardUtilizationRate

# urutan HARUS sama seperti training
sample = np.array([[
    payment_ratio,
    debt_income_ratio,
    credit_utilization,
    PreviousLoanDefaults,
    PaymentHistory
]])

# =========================
# CLUSTER PREDICTION
# =========================
sample_scaled = scaler.transform(sample)
cluster = kmeans.predict(sample_scaled)

# =========================
# RISK PREDICTION
# =========================
risk_pred = risk_model.predict(sample)
risk_prob = risk_model.predict_proba(sample)

print("Cluster:", cluster[0])
print("Default Risk Prediction:", risk_pred[0])
print("Default Probability:", risk_prob[0][1])
