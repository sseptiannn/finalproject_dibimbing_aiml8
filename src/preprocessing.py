import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def load_data(self, path: str):
        return pd.read_csv(path)

    def feature_engineering(self, df: pd.DataFrame):
        df = df.copy()

        # Behavior-based features
        df["payment_ratio"] = df["MonthlyLoanPayment"] / df["MonthlyIncome"]
        df["debt_income_ratio"] = df["TotalDebtToIncomeRatio"]
        df["credit_utilization"] = df["CreditCardUtilizationRate"]

        return df

    def select_features_for_clustering(self, df):
        features = [
            "payment_ratio",
            "debt_income_ratio",
            "credit_utilization",
            "PreviousLoanDefaults",
            "PaymentHistory"
        ]
        return df[features]

    def scale(self, X):
        return self.scaler.fit_transform(X)
