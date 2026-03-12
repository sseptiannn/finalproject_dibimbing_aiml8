import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score

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
    # LOAD DATA
    # =========================
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(data_path)

    print("Data shape:", df.shape)

    # =========================
    # FEATURE ENGINEERING
    # =========================
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

        print(f"\nTraining clustering model: {name}")

        labels = model.train(X_scaled)

        sil_score = ModelEvaluation.evaluate_clustering(X_scaled, labels)

        print(f"{name} Silhouette Score: {sil_score:.4f}")

        joblib.dump(
            model.model,
            os.path.join(model_path, f"clustering_model_{name}.pkl")
        )

        if sil_score > best_cluster_score:
            best_cluster_score = sil_score
            best_cluster_name = name
            best_labels = labels

    print(f"\nBest clustering model: {best_cluster_name}")

    df["Cluster"] = best_labels

    joblib.dump(preprocessor.scaler, os.path.join(model_path, "scaler.pkl"))

    print("Clustering models & scaler saved")

    # =========================
    # RISK MODEL SECTION
    # =========================
    print("\nTRAINING RISK MODELS")

    # IMPORTANT: remove target leakage
    risk_features = [
        "MonthlyLoanPayment",
        "MonthlyIncome",
        "TotalDebtToIncomeRatio",
        "CreditCardUtilizationRate",
        "PaymentHistory",
        "Cluster"
    ]

    X_risk = df[risk_features]

    # target
    y_risk = df["PreviousLoanDefaults"]

    # =========================
    # TRAIN TEST SPLIT
    # =========================
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

        # =========================
        # CROSS VALIDATION
        # =========================
        cv_scores = cross_val_score(
            risk_model.model,
            X_train,
            y_train,
            cv=5,
            scoring="roc_auc"
        )

        print("Cross Validation ROC-AUC:", cv_scores)
        print("Mean CV ROC-AUC:", cv_scores.mean())

        # =========================
        # TRAIN MODEL
        # =========================
        risk_model.train(X_train, y_train)

        # =========================
        # PREDICTION
        # =========================
        y_pred = risk_model.predict(X_test)
        y_prob = risk_model.predict_proba(X_test)

        report, roc_score = ModelEvaluation.evaluate_classification(
            y_test,
            y_pred,
            y_prob
        )

        print(report)
        print(f"{m} Test ROC-AUC: {roc_score:.4f}")

        joblib.dump(
            risk_model.model,
            os.path.join(model_path, f"risk_model_{m}.pkl")
        )

        if roc_score > best_score:
            best_score = roc_score
            best_model_name = m

    print(f"\nBest risk model: {best_model_name}")

    print("All risk models saved")

    print("\nTRAINING COMPLETED")


if __name__ == "__main__":
    main()