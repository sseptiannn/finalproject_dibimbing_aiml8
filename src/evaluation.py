from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    silhouette_score
)

class ModelEvaluation:

    @staticmethod
    def evaluate_clustering(X, labels):
        return silhouette_score(X, labels)

    @staticmethod
    def evaluate_classification(y_true, y_pred, y_prob):
        report = classification_report(y_true, y_pred)
        roc = roc_auc_score(y_true, y_prob)
        return report, roc
