# src/clustering_gmm.py
from sklearn.mixture import GaussianMixture

class CustomerClusteringGMM:
    def __init__(self, n_clusters=3, random_state=42):
        self.model = GaussianMixture(
            n_components=n_clusters,
            random_state=random_state
        )

    def train(self, X):
        self.model.fit(X)
        labels = self.model.predict(X)
        return labels