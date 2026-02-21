# src/risk_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class RiskModel:
    def __init__(self, model_type="rf"):
        if model_type == "rf":
            self.model = RandomForestClassifier(random_state=42)

        elif model_type == "logreg":
            self.model = LogisticRegression(max_iter=1000)

        elif model_type == "xgb":
            self.model = XGBClassifier(
                eval_metric="logloss",
                random_state=42
            )
        else:
            raise ValueError("Unknown model_type")

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]