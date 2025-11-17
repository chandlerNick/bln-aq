import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from chronos import Chronos2Pipeline
from src.spatiots.pipeline import SpatioTemporalPipeline
from src.spatiots.temporal.chronos2 import Chronos2Wrapper
from src.spatiots.spatial.kriging import OrdinaryKrigingModel


# =============================
# Load Chronos2
# =============================
chronos2 = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda"
)

temporal_model = Chronos2Wrapper(
    pipeline=chronos2,
    id_column="id",
    time_column="timestamp",
    target="target"
)


# =============================
# Fake data for a demo
# =============================
def fake_ts():
    base = np.sin(np.linspace(0, 6, 48)) * 4 + 20
    noise = np.random.randn(48)
    return base + noise

lats = np.random.uniform(52.0, 53.0, size=20)
lons = np.random.uniform(13.0, 14.0, size=20)

records = []
for lat, lon in zip(lats, lons):
    ts = fake_ts()
    for i, v in enumerate(ts):
        records.append({
            "lat": lat,
            "lon": lon,
            "timestamp": i,
            "target": v,
        })

df = pd.DataFrame(records)


# =============================
# Build pipeline
# =============================
spatial_model = OrdinaryKrigingModel(variogram="exponential")

pipe = SpatioTemporalPipeline(
    temporal_model=temporal_model,
    spatial_model=spatial_model
)

# =============================
# Forecast
# =============================
horizon = 24
df_fore = pipe.forecast_per_location(df, horizon=horizon)

# Krige the forecast for horizon index 10
X, Y, Z = pipe.spatial_interpolate(df_fore, horizon_idx=10, grid_size=120)

# =============================
# Plot result
# =============================
plt.figure(figsize=(7,6))
plt.pcolormesh(X, Y, Z, shading="auto")
plt.scatter(df_fore["lon"].unique(), df_fore["lat"].unique(), c="k", s=20)
plt.title("Chronos2 + Ordinary Kriging (Horizon = 10)")
plt.colorbar(label="Forecast")
plt.savefig("fig.pdf")
