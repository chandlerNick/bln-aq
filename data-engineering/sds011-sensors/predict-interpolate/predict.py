#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pyarrow.dataset as ds
from pathlib import Path
from chronos import Chronos2Pipeline
import numpy as np
from pykrige.ok import OrdinaryKriging
import os
from pyproj import Transformer


# =====================================================================
# Paths
# =====================================================================

parquet_dir = Path(os.environ.get("PARQUET_DIR", "/data/parquet"))
output_dir = parquet_dir
output_dir.mkdir(exist_ok=True, parents=True)

BBOX = {
    "lat_min": 52.3383,
    "lat_max": 52.6755,
    "lon_min": 13.0884,
    "lon_max": 13.7612,
}


# =====================================================================
# Load dataset
# =====================================================================

dataset = ds.dataset(parquet_dir / "sds011.parquet", format="parquet")
df = dataset.to_table().to_pandas()

# Basic cleaning
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["P2"] = pd.to_numeric(df["P2"], errors="coerce")
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

df = df.dropna(subset=["timestamp", "P2", "lat", "lon"])

df["date"] = df["timestamp"].dt.floor("D")

daily_avg = (
    df.groupby(["lat", "lon", "date", "sensor_id"], as_index=False)["P2"]
      .mean()
      .rename(columns={"P2": "target"})
)
daily_avg = daily_avg.dropna(subset=["target"]).reset_index(drop=True)

daily_avg["item_id"] = daily_avg["sensor_id"]
location_dict = (
    daily_avg.groupby("item_id")[["lat","lon"]]
      .first()
      .astype(float)
      .apply(tuple, axis=1)
      .to_dict()
)

daily_df = daily_avg[["date", "item_id", "target"]].rename(columns={"date": "timestamp"})


# =====================================================================
# Chronos forecasting
# =====================================================================

FORECAST_DAYS = 3
QUANTILES = [0.1, 0.9]

df_wide = daily_df.pivot(index="timestamp", columns="item_id", values="target").asfreq("D")

valid_sensors = df_wide.notna().sum()[lambda x: x >= 10].index
df_wide = df_wide[valid_sensors]

train_long = df_wide.reset_index().melt(id_vars="timestamp", var_name="item_id", value_name="target")

pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

pred_df = pipeline.predict_df(
    train_long,
    prediction_length=FORECAST_DAYS,
    quantile_levels=QUANTILES,
    id_column="item_id",
    timestamp_column="timestamp",
    target="target"
)

# Add location fields
pred_df[["lat", "lon"]] = pred_df["item_id"].map(lambda x: location_dict.get(x, (np.nan, np.nan))).apply(pd.Series)
pred_df = pred_df.dropna(subset=["lat", "lon"])
pred_df["lat"] = pred_df["lat"].astype(float)
pred_df["lon"] = pred_df["lon"].astype(float)

pred_df = pred_df.rename(columns={"predictions": "value"})
pred_df["is_sensor"] = True

data = pred_df.drop(columns=["target_name"]).copy()

# Filtering
latest_ts = data["timestamp"].max()
data = data[data["timestamp"] == latest_ts]

data = data[(data["value"] >= 0) & ((data["0.9"] - data["0.1"]) <= 100)]
upper_cutoff = data["value"].quantile(0.98)
data = data[data["value"] <= upper_cutoff]

# Clip extremes
low, high = np.percentile(data["value"], [1, 99])
data["value"] = data["value"].clip(low, high)

save_date = pd.to_datetime(latest_ts).normalize()  # ensure datetime not date

# Save first result
clean_file = output_dir / f"predicted_data_{save_date.date()}.parquet"
data.to_parquet(clean_file, index=False)
print(f"Saved cleaned prediction data to {clean_file}")


# =====================================================================
# Kriging (in projected meters)
# =====================================================================

transformer = Transformer.from_crs("epsg:4326", "epsg:25833", always_xy=True)

# Explicit float64 to avoid PyArrow errors
lon_arr = data["lon"].astype(float).values
lat_arr = data["lat"].astype(float).values
vals_arr = data["value"].astype(float).values

x, y = transformer.transform(lon_arr, lat_arr)

x = np.asarray(x, dtype=float)
y = np.asarray(y, dtype=float)
vals_arr = np.asarray(vals_arr, dtype=float)

# Grid
buffer_m = 1000
x_min, x_max = x.min() - buffer_m, x.max() + buffer_m
y_min, y_max = y.min() - buffer_m, y.max() + buffer_m

nx, ny = 200, 200
x_grid = np.linspace(x_min, x_max, nx)
y_grid = np.linspace(y_min, y_max, ny)

OK = OrdinaryKriging(
    x, y, vals_arr,
    variogram_model="spherical",
    verbose=False,
    enable_plotting=False
)

Z, ss = OK.execute("grid", x_grid, y_grid)

Z = np.asarray(Z.filled(np.nan), dtype=float)
ss = np.asarray(ss.filled(np.nan), dtype=float)

Xg, Yg = np.meshgrid(x_grid, y_grid)
lon_grid, lat_grid = transformer.transform(Xg.ravel(), Yg.ravel(), direction="INVERSE")

# =====================================================================
# Build kriged DataFrame
# =====================================================================

kriged_df = pd.DataFrame({
    "item_id": np.nan,
    "timestamp": save_date,
    "value": Z.ravel().astype(float),
    "0.1": (Z - 1.2816 * np.sqrt(ss)).ravel().astype(float),
    "0.9": (Z + 1.2816 * np.sqrt(ss)).ravel().astype(float),
    "lat": lat_grid.astype(float),
    "lon": lon_grid.astype(float),
    "is_sensor": False
})

# =====================================================================
# Combine final DF
# =====================================================================

data["timestamp"] = pd.to_datetime(data["timestamp"])
kriged_df["timestamp"] = pd.to_datetime(kriged_df["timestamp"])

full_df = pd.concat([data, kriged_df], ignore_index=True)

# CLEAN types
full_df["lat"] = pd.to_numeric(full_df["lat"], errors="coerce").astype(float)
full_df["lon"] = pd.to_numeric(full_df["lon"], errors="coerce").astype(float)
full_df["value"] = pd.to_numeric(full_df["value"], errors="coerce").astype(float)
full_df["0.1"] = pd.to_numeric(full_df["0.1"], errors="coerce").astype(float)
full_df["0.9"] = pd.to_numeric(full_df["0.9"], errors="coerce").astype(float)
full_df["is_sensor"] = full_df["is_sensor"].astype(bool)

# =====================================================================
# Final parquet write
# =====================================================================

final_file = parquet_dir / f"predicted_plus_kriged_{save_date.date()}.parquet"
full_df.to_parquet(final_file, index=False)

print(f"Saved combined sensor + kriged data to {final_file}")
