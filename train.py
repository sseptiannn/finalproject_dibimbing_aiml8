import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import DataPreprocessor
from src.clustering_kmeans import CustomerClusteringKMeans
from src.clustering_gmm import CustomerClusteringGMM
from src.risk_model import RiskModel
from src.evaluation import ModelEvaluation


def main():

    print("LOAD DATA & PREPROCESSING")
    data_path = "data/loan_data.csv"
    model_path = "models"
    os.makedirs(model_path, exist_ok=True)

    # =========================
    # LOAD & FEATURE ENGINEERING
    # =========================
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(data_path)
    print("Data shape:", df.shape)

    print("\nFEATURE ENGINEERING")
    df = preprocessor.feature_engineering(df)

    # =========================
    # CLUSTERING SECTION
    # =========================
    print("\nTRAINING CLUSTERING MODELS")

    X_cluster = preprocessor.select_features_for_clustering(df)
    X_scaled = preprocessor.scale(X_cluster)

    clustering_models = {
        "kmeans": CustomerClusteringKMeans(n_clusters=2),
        "gmm": CustomerClusteringGMM(n_clusters=3),
    }

    best_cluster_score = -1
    best_cluster_name = None
    best_labels = None

    for name, model in clustering_models.items():
        print(f"\nTraining clustering: {name}")
        labels = model.train(X_scaled)
        sil_score = ModelEvaluation.evaluate_clustering(X_scaled, labels)
        print(f"{name} Silhouette Score: {sil_score:.4f}")

        # Save each clustering model separately
        joblib.dump(model.model, os.path.join(model_path, f"clustering_model_{name}.pkl"))

        # Track best model
        if sil_score > best_cluster_score:
            best_cluster_score = sil_score
            best_cluster_name = name
            best_labels = labels

    print(f"\nBest clustering model: {best_cluster_name}")
    df["Cluster"] = best_labels

    # Save scaler once
    joblib.dump(preprocessor.scaler, os.path.join(model_path, "scaler.pkl"))
    print("Clustering models & scaler saved.")

    # =========================
    # RISK MODEL SECTION
    # =========================
    print("\nTRAINING RISK MODELS")
    X_risk = X_cluster
    y_risk = df["PreviousLoanDefaults"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_risk,
        y_risk,
        test_size=0.2,
        random_state=42,
        stratify=y_risk
    )

    models_to_try = ["logreg", "rf", "xgb"]

    best_score = 0
    best_model_name = None

    for m in models_to_try:
        print(f"\nTraining risk model: {m}")
        risk_model = RiskModel(model_type=m)
        risk_model.train(X_train, y_train)

        y_pred = risk_model.predict(X_test)
        y_prob = risk_model.predict_proba(X_test)

        report, roc_score = ModelEvaluation.evaluate_classification(y_test, y_pred, y_prob)
        print(report)
        print(f"{m} ROC-AUC: {roc_score:.4f}")

        # Save each risk model separately
        joblib.dump(risk_model.model, os.path.join(model_path, f"risk_model_{m}.pkl"))

        # Track best model
        if roc_score > best_score:
            best_score = roc_score
            best_model_name = m

    print(f"\nBest risk model: {best_model_name}")
    print("All risk models saved.")
    print("\nTRAINING COMPLETED")


if __name__ == "__main__":
    main()