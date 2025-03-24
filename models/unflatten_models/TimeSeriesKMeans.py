from typing import *

import numpy as np
from tslearn.metrics import dtw
from tslearn.metrics import cdist_dtw
from sklearn.metrics import silhouette_score
from tslearn.clustering import TimeSeriesKMeans

from models.cluster import ClusterModel

class TimeSeries_KMeans(ClusterModel):
    def __init__(self, n_clusters: int):
        self.model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw")
        self.model_name = "time_series_kmeans"
    

    def fit(self, X: np.ndarray) -> Any:
        self.model.fit(X)
        return self


    def fit_predict(self, X):
        labels = self.model.fit_predict(X)
        return labels


    def predict(self, X):
        labels = self.model.predict(X)
        return labels