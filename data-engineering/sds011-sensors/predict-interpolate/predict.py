#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pyarrow.dataset as ds
from pathlib import Path
from chronos import Chronos2Pipeline
import numpy as np
from pykrige.ok import OrdinaryKriging
import os


# ----------------------------
# Paths
# ----------------------------

BBOX = {
    "lat_min": 52.3383,
    "lat_max": 52.6755,
    "lon_min": 13.0884,
    "lon_max": 13.7612,
}

parquet_dir = Path(os.environ.get("PARQUET_DIR", "/data/parquet"))
output_dir = parquet_dir
output_dir.mkdir(exist_ok=True, parents=True)


# ----------------------------
# Load dataset
# ----------------------------
dataset = ds.dataset(parquet_dir / "sds011.parquet", format="parquet")
df = dataset.to_table().to_pandas()

# ----------------------------
# Preprocess and aggregate
# ----------------------------

print(df.head(5), df.columns)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.floor('D')
df['P2'] = pd.to_numeric(df['P2'], errors='coerce')

daily_avg = (
    df.groupby(['lat', 'lon', 'date', 'sensor_id'], as_index=False)['P2']
      .mean()
      .rename(columns={'P2': 'target'})
)
daily_avg = daily_avg.dropna(subset=['target']).reset_index(drop=True)

daily_avg['item_id'] = daily_avg['sensor_id']
location_dict = daily_avg.groupby('item_id')[['lat','lon']].first().apply(tuple, axis=1).to_dict()
daily_df = daily_avg[['date', 'item_id', 'target']].rename(columns={'date':'timestamp'})

# ----------------------------
# Chronos preparation
# ----------------------------
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

# ----------------------------
# Prepare for saving
# ----------------------------
pred_df[['lat', 'lon']] = pred_df['item_id'].map(lambda x: location_dict.get(x, (None, None))).apply(pd.Series)
pred_df = pred_df.dropna(subset=['lat','lon'])
pred_df = pred_df.rename(columns={"predictions": "value"})
pred_df["is_sensor"] = True

data = pred_df.drop(columns=['target_name']).copy()

# Filter unrealistic predictions
latest_ts = data['timestamp'].max()
threshold = 100

data = data[(data['value'] >= 0) & ((data['0.9'] - data['0.1']) <= threshold)]
data = data[data['timestamp'] == latest_ts]
upper_cutoff = data['value'].quantile(0.98)
data = data[data['value'] <= upper_cutoff]

lower_val, upper_val = np.percentile(data['value'], [1, 99])
data['value'] = data['value'].clip(lower=lower_val, upper=upper_val)

# ----------------------------
# Save final DataFrame
# ----------------------------
save_date = latest_ts.date()
output_file = output_dir / f"predicted_data_{save_date}.parquet"
data.to_parquet(output_file, index=False)
print(f"Saved cleaned prediction data to {output_file}")


### KRIGING

# ----------------------------
# Kriging setup
# ----------------------------
lon, lat, z = data["lon"].astype(float).values, data["lat"].astype(float).values, data["value"].astype(float).values

buffer = 0.01
lon_grid = np.linspace(BBOX['lon_min'] - buffer, BBOX['lon_max'] + buffer, 200)
lat_grid = np.linspace(BBOX['lat_min'] - buffer, BBOX['lat_max'] + buffer, 200)
Lon, Lat = np.meshgrid(lon_grid, lat_grid)

OK = OrdinaryKriging(lon, lat, z, variogram_model='hole-effect', verbose=False, enable_plotting=False)
Z_kriged, ss = OK.execute('grid', lon_grid, lat_grid)

Z_kriged = Z_kriged.filled(np.nan)
ss = ss.filled(np.nan)

np.save(parquet_dir / f"Z_kriged_{save_date}.npy", Z_kriged)
np.save(parquet_dir / f"kriging_variance_{save_date}.npy", ss)
