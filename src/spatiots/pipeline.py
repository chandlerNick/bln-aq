# Two stage spatio-temporal forecasting pipeline
# 17.11.25 - Nick Chandler
import numpy as np
import pandas as pd

class SpatioTemporalPipeline:
    """
    Generic two-stage forecasting pipeline:
    1. Temporal forecasting per location
    2. Spatial interpolation of forecasts
    """
    def __init__(self, temporal_model, spatial_model):
        self.temporal_model = temporal_model
        self.spatial_model = spatial_model

    def forecast_per_location(self, df, horizon, value_col="target"):
        grouped = df.groupby(["lat", "lon"])
        outputs = []

        for (lat, lon), group in grouped:
            ts = group.sort_values("timestamp")[value_col].values
            preds = self.temporal_model.predict(ts, horizon)

            for h in range(horizon):
                outputs.append({
                    "lat": lat,
                    "lon": lon,
                    "horizon": h,
                    "forecast": preds[h]
                })

        return pd.DataFrame(outputs)

    def spatial_interpolate(self, df_forecast, horizon_idx=0, grid_size=100):
        df_h = df_forecast[df_forecast["horizon"] == horizon_idx]

        coords = df_h[["lon", "lat"]].values
        values = df_h["forecast"].values

        # fit spatial model
        self.spatial_model.fit(coords, values)

        # generate grid
        xs = np.linspace(coords[:, 0].min(), coords[:, 0].max(), grid_size)
        ys = np.linspace(coords[:, 1].min(), coords[:, 1].max(), grid_size)
        X, Y = np.meshgrid(xs, ys)

        # spatial prediction
        Z = self.spatial_model.predict_grid(X, Y)

        return X, Y, Z
