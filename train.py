import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import DataPreprocessor
from src.clustering import CustomerClustering
from src.risk_model import RiskModel
from src.evaluation import ModelEvaluation


def main():

    print("========== LOADING DATA ==========")

    data_path = "data/loan_data.csv"
    model_path = "models"

    os.makedirs(model_path, exist_ok=True)

    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(data_path)

    print("Data shape:", df.shape)

    # =========================
    # FEATURE ENGINEERING
    # =========================
    print("\n========== FEATURE ENGINEERING ==========")
    df = preprocessor.feature_engineering(df)

    # =========================
    # CLUSTERING
    # =========================
    print("\n========== TRAINING CLUSTERING MODEL ==========")

    X_cluster = preprocessor.select_features_for_clustering(df)
    X_scaled = preprocessor.scale(X_cluster)

    clustering_model = CustomerClustering(n_clusters=3)
    cluster_labels = clustering_model.train(X_scaled)

    df["Cluster"] = cluster_labels

    sil_score = ModelEvaluation.evaluate_clustering(X_scaled, cluster_labels)
    print(f"Silhouette Score: {sil_score:.4f}")

    # Save clustering model & scaler
    joblib.dump(clustering_model.model, os.path.join(model_path, "kmeans.pkl"))
    joblib.dump(preprocessor.scaler, os.path.join(model_path, "scaler.pkl"))

    print("Clustering model saved.")

    # =========================
    # RISK MODEL (SUPERVISED)
    # =========================
    print("\n========== TRAINING RISK MODEL ==========")

    X_risk = X_cluster
    y_risk = df["PreviousLoanDefaults"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_risk, y_risk,
        test_size=0.2,
        random_state=42,
        stratify=y_risk
    )

    risk_model = RiskModel()
    risk_model.train(X_train, y_train)

    y_pred = risk_model.predict(X_test)
    y_prob = risk_model.predict_proba(X_test)

    report, roc_score = ModelEvaluation.evaluate_classification(
        y_test, y_pred, y_prob
    )

    print("\nClassification Report:")
    print(report)
    print(f"ROC-AUC Score: {roc_score:.4f}")

    joblib.dump(risk_model.model, os.path.join(model_path, "risk_model.pkl"))

    print("Risk model saved.")

    print("\n========== TRAINING COMPLETED SUCCESSFULLY ==========")


if __name__ == "__main__":
    main()
