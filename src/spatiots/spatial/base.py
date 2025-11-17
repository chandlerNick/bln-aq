from abc import ABC, abstractmethod
import numpy as np

class SpatialModel(ABC):
    @abstractmethod
    def fit(self, coords: np.ndarray, values: np.ndarray):
        pass

    @abstractmethod
    def predict(self, coords: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_grid(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        pass
