from typing import *

import numpy as np
from sklearn.cluster import HDBSCAN

from models.cluster import ClusterModel

class Flatten_HDBSCAN(ClusterModel):
    def __init__(self):
        self.model = HDBSCAN()
        self.model_name = "hdbscan"
    

    def fit(self, X: np.ndarray) -> Any:
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

        