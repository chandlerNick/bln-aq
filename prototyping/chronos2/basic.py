#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from chronos import Chronos2Pipeline
from sklearn.metrics import mean_squared_error
import time

# ----------------------------
# Config
# ----------------------------
PARQUET_PATH = "/storage/bln-aq/data/2024-citsci-pollutants-hourly.parquet"
PREDICTIONS_OUTPUT = "/storage/bln-aq/data/chronos-predictions-2024.csv"


# ----------------------------
# Helper functions
# ----------------------------
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def regularize_daily(df):
    """Aggregate hourly to daily, regularize, and fill missing item_id."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Aggregate to daily mean per sensor
    daily_df = df.groupby(['item_id', pd.Grouper(key='timestamp', freq='D')])['target'].mean()
    daily_df = daily_df.reset_index()

    # Regularize each sensor
    regularized = []
    for sensor in daily_df['item_id'].unique():
        sensor_df = daily_df[daily_df['item_id'] == sensor].set_index('timestamp')
        full_idx = pd.date_range(sensor_df.index.min(), sensor_df.index.max(), freq='D')
        sensor_df = sensor_df.reindex(full_idx)
        sensor_df = sensor_df.rename_axis('timestamp').reset_index()
        sensor_df['item_id'] = sensor
        regularized.append(sensor_df)

    return pd.concat(regularized, ignore_index=True)

# ----------------------------
# Load and preprocess data
# ----------------------------
df = pd.read_parquet(PARQUET_PATH)

# Aggregate PM2.5 per sensor/location + hour
df['timestamp_hour'] = pd.to_datetime(df['timestamp_hour'])
df = df.groupby(['lat', 'lon', 'timestamp_hour'], as_index=False)['PM2_5'].mean()

# Create a unique key for each sensor location
unique_coords = df.drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
unique_coords["loc_id"] = range(1, len(unique_coords) + 1)
df = df.merge(unique_coords, on=["lat", "lon"], how="left")

# Drop unnecessary _y columns and rename _x
df = df.drop(columns=[c for c in df.columns if c.endswith("_y")])
df = df.rename(columns={c: c[:-2] for c in df.columns if c.endswith("_x")})

# Map loc_id to coordinates
loc_dict = df.groupby("loc_id")[["lat","lon"]].first().apply(tuple, axis=1).to_dict()



# Remove lat/lon and rename columns for Chronos2
df = df.drop(columns=["lat","lon"]).rename(columns={
    "timestamp_hour": "timestamp",
    "loc_id": "item_id",
    "PM2_5": "target"
})

# ----------------------------
# Daily aggregation & regularization
# ----------------------------
df = regularize_daily(df)

# Remove sensors with fewer than 10 data points
sensor_counts = df['item_id'].value_counts()
valid_sensors = sensor_counts[sensor_counts >= 10].index
df = df[df['item_id'].isin(valid_sensors)]

# Remove problematic sensor if needed
df = df[df['item_id'] != 163]

# Save the loc_dict (mapping loc_id → (lat, lon))
import json
loc_dict_path = "/storage/bln-aq/data/loc_dict.json"
with open(loc_dict_path, "w") as f:
    # Only save locs that remain after filtering
    filtered_loc_dict = {k: v for k, v in loc_dict.items() if k in valid_sensors and k != 163}
    json.dump(filtered_loc_dict, f)

# Save the daily-regularized data
df.to_parquet("/storage/bln-aq/data/2024-citsci-pm2.5-daily.parquet")

# ----------------------------------------------------------------------------------------------------------------------------------------------
# Get preds - could be a separate file
# ----------------------------------------------------------------------------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Generate predictions each step in a sliding window 

# ----------------------------
# Config
# ----------------------------
WINDOW_DAYS = 60
FORECAST_DAYS = 3
QUANTILES = [0.1, 0.5, 0.9]

PREDICTIONS_OUTPUT = "/storage/bln-aq/data/chronos_daily_forecast_sliding.csv"

pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df_wide = df.pivot(index="timestamp", columns="item_id", values="target").asfreq("D")

# Only keep sensors with enough data
sensor_counts = df_wide.notna().sum()
valid_sensors = sensor_counts[sensor_counts >= 10].index
df_wide = df_wide[valid_sensors]

# ----------------------------
# Sliding-window prediction
# ----------------------------
dates = df_wide.index
preds = []
batch_id = 0

start_time = time.time()  # <--- start timer

for t_idx in range(WINDOW_DAYS, len(dates)-FORECAST_DAYS+1):
    print(f"batch: {batch_id}")
    batch_id += 1
    train_window = df_wide.iloc[t_idx-WINDOW_DAYS:t_idx]

    # Convert back to long format
    train_long = train_window.reset_index().melt(id_vars="timestamp", var_name="item_id", value_name="target")

    # Chronos prediction
    pred_df = pipeline.predict_df(
        train_long,
        prediction_length=FORECAST_DAYS,
        quantile_levels=QUANTILES,
        id_column="item_id",
        timestamp_column="timestamp",
        target="target"
    )

    # Assign batch_id
    pred_df["batch_id"] = batch_id
    preds.append(pred_df)

end_time = time.time()  # <--- end timer
elapsed = end_time - start_time
print(f"Sliding-window predictions completed in {elapsed:.2f} seconds")  # Sliding-window predictions completed in 128.43 seconds

# ----------------------------
# Combine all batches
# ----------------------------
all_preds = pd.concat(preds, ignore_index=True)
all_preds.to_csv(PREDICTIONS_OUTPUT, index=False)
print(f"Sliding-window predictions saved to {PREDICTIONS_OUTPUT}")