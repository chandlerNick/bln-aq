from abc import ABC, abstractmethod
import numpy as np

class TemporalModel(ABC):
    @abstractmethod
    def predict(self, ts: np.ndarray, horizon: int) -> np.ndarray:
        """
        Return an array of length horizon containing forecasts.
        """
        pass
