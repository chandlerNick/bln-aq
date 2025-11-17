import numpy as np
from pykrige.ok import OrdinaryKriging
from .base import SpatialModel

class OrdinaryKrigingModel(SpatialModel):
    def __init__(self, variogram='spherical'):
        self.variogram = variogram

    def fit(self, coords, values):
        xs, ys = coords[:, 0], coords[:, 1]
        self.krige = OrdinaryKriging(
            xs, ys, values,
            variogram_model=self.variogram,
            enable_plotting=False,
            verbose=False
        )

    def predict(self, coords):
        xs, ys = coords[:,0], coords[:,1]
        z, ss = self.krige.execute("points", xs, ys)
        return z

    def predict_grid(self, X, Y):
        z, ss = self.krige.execute("grid", X[0], Y[:, 0])
        return z
