from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class CustomerClusteringKMeans:
    def __init__(self, n_clusters=3):
        self.model = KMeans(n_clusters=n_clusters, random_state=42)

    def train(self, X):
        labels = self.model.fit_predict(X)
        return labels

    def evaluate(self, X, labels):
        return silhouette_score(X, labels)
