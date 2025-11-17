import pandas as pd
import numpy as np
from .base import TemporalModel

class Chronos2Wrapper(TemporalModel):
    """
    Wrapper for Amazon Chronos-2 foundation model.
    Assumes you call Chronos2Pipeline.from_pretrained(...) externally
    and pass the resulting pipeline object here.
    """

    def __init__(self, pipeline, id_column="id", time_column="timestamp", target="target"):
        self.pipeline = pipeline
        self.id_column = id_column
        self.time_column = time_column
        self.target = target

    def predict(self, ts: np.ndarray, horizon: int) -> np.ndarray:
        """
        NOTE: ts must come in as a numpy array for a *single* location.
        We convert it into a small DataFrame so Chronos2Pipeline can process it.

        Forecasts only the median (0.5 quantile) unless you add quantiles.
        """

        # Build context DataFrame for Chronos2
        df = pd.DataFrame({
            self.id_column: [0] * len(ts),
            self.time_column: np.arange(len(ts)),
            self.target: ts
        })

        pred_df = self.pipeline.predict_df(
            df,
            future_df=None,
            prediction_length=horizon,
            quantile_levels=[0.5],
            id_column=self.id_column,
            timestamp_column=self.time_column,
            target=self.target,
        )

        # Chronos2 returns a MultiIndex DataFrame with quantiles as columns
        median_name = "0.5"
        yhat = pred_df[median_name].values

        return yhat
