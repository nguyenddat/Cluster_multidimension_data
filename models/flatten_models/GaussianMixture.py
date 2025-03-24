from sklearn.mixture import GaussianMixture

from models.cluster import ClusterModel

class Gaussian_Mixture(ClusterModel):
    def __init__(self):
        self.model = GaussianMixture(n_components=8, covariance_type='full', random_state=42)
        self.model_name = "gaussian_mixture"
    

    def fit(self, X):
        self.model.fit(X)
        return self


    def fit_predict(self, X):
        labels = self.model.fit_predict(X)
        return labels
    
    
    def predict(self, X):
        labels = self.model.predict(X)
        return labels