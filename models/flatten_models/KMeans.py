import numpy as np
from sklearn.cluster import KMeans

from models.cluster import ClusterModel


class Flatten_KMeans(ClusterModel):
    def __init__(
            self,
            n_clusters: int,
            max_iter: int = 100
    ):
        self.model = KMeans(n_clusters = n_clusters, max_iter = max_iter)
        self.model_name = "kmeans"

    
    def fit(self, X: np.ndarray):
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        self.model.fit(X)
        return self


    def fit_predict(self, X):
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        labels = self.model.fit_predict(X)
        return labels
    

    def predict(self, X):
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        labels = self.model.predict(X)
        return labels
        
        
    
    


