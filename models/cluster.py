from typing import *

import numpy as np
from abc import ABC, abstractmethod

class ClusterModel(ABC):
    model: Any
    model_name: str

    @abstractmethod
    def fit(self, X: np.ndarray) -> Any:
        raise NotImplementedError()

    
    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()