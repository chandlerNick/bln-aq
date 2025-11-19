#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import pandas as pd
import pyarrow.dataset as ds
from pathlib import Path
from chronos import Chronos2Pipeline
import numpy as np
from pykrige.ok import OrdinaryKriging

# Path to the directory
parquet_dir = Path("tester.parquet")

dataset = ds.dataset(parquet_dir, format="parquet")
df = dataset.to_table().to_pandas()
df.head()


# In[2]:


# ----------------------------
# Convert timestamp to datetime
# ----------------------------
df['timestamp'] = pd.to_datetime(df['timestamp'])

# ----------------------------
# Aggregate P2 by sensor location and day
# ----------------------------
df['date'] = df['timestamp'].dt.floor('D')

# Ensure P2 is float before aggregation
df['P2'] = pd.to_numeric(df['P2'], errors='coerce')  # invalid parsing becomes NaN

# Create date column for daily aggregation
df['date'] = pd.to_datetime(df['timestamp']).dt.floor('D')

# Aggregate by sensor location and day
daily_avg = (
    df.groupby(['lat', 'lon', 'date', 'sensor_id'], as_index=False)['P2']
      .mean()
      .rename(columns={'P2': 'target'})  # Chronos expects 'target'
)

# Drop any rows where aggregation produced NaN (if all values were NaN that day)
daily_avg = daily_avg.dropna(subset=['target']).reset_index(drop=True)

# ---------------------------
print(daily_avg.head())

# Use sensor_id as item_id
daily_avg['item_id'] = daily_avg['sensor_id']

# Build mapping for kriging
location_dict = daily_avg.groupby('item_id')[['lat','lon']].first().apply(tuple, axis=1).to_dict()

# Keep only relevant columns -- CHRONOS DF
daily_df = daily_avg[['date', 'item_id', 'target']].rename(columns={'date':'timestamp'})

print(daily_df.columns)



# In[3]:


# ----------------------------
# Config
# ----------------------------
FORECAST_DAYS = 3
QUANTILES = [0.1, 0.5, 0.9]

# Ensure correct columns for Chronos
daily_df = daily_df.rename(columns={"sensor_id":"item_id", "date":"timestamp"})
daily_df['timestamp'] = pd.to_datetime(daily_df['timestamp'])

# ----------------------------
# Pivot to wide format if needed
# ----------------------------
df_wide = daily_df.pivot(index="timestamp", columns="item_id", values="target").asfreq("D")

# Filter sensors with enough data
sensor_counts = df_wide.notna().sum()
valid_sensors = sensor_counts[sensor_counts >= 10].index
df_wide = df_wide[valid_sensors]

# ----------------------------
# Convert back to long format for Chronos
# ----------------------------
train_long = df_wide.reset_index().melt(id_vars="timestamp", var_name="item_id", value_name="target")

# ----------------------------
# Load Chronos model
# ----------------------------
pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

# ----------------------------
# Predict
# ----------------------------
pred_df = pipeline.predict_df(
    train_long,
    prediction_length=FORECAST_DAYS,
    quantile_levels=QUANTILES,
    id_column="item_id",
    timestamp_column="timestamp",
    target="target"
)


# In[4]:


pred_df.head()


# In[5]:


# pred_df is your predictions DataFrame
# location_dict maps item_id -> (lat, lon)

# 1. Map lat/lon from location_dict
pred_df[['lat', 'lon']] = pred_df['item_id'].map(lambda x: location_dict.get(x, (None, None))).apply(pd.Series)

# 2. Optionally drop rows where mapping failed
pred_df = pred_df.dropna(subset=['lat','lon'])

# 3. Rename prediction column to 'value' for clarity in kriging script
pred_df = pred_df.rename(columns={"predictions": "value"})

# 4. Add is_sensor flag
pred_df["is_sensor"] = True

# Now pred_df has exactly the columns your kriging script expects:
# ['item_id', 'timestamp', 'target_name', 'value', '0.1', '0.5', '0.9', 'lat', 'lon', 'is_sensor']
# You can pass it as `data` in your kriging snippet
data = pred_df.drop(columns=['target_name', '0.5']).copy()

# Winsorize the data
# Define lower/upper percentiles
lower_quantl = 1  # 1st percentile
upper_quantl = 99  # 99th percentile

# Compute the actual values at those percentiles
lower_val = np.percentile(data['value'], lower_quantl)
upper_val = np.percentile(data['value'], upper_quantl)

# Winsorize the column in-place
data['value'] = data['value'].clip(lower=lower_val, upper=upper_val)

data.head()

data.head()


# In[6]:


# 0. Set up constants
BBOX = {
    "lat_min": 52.3383,
    "lat_max": 52.6755,
    "lon_min": 13.0884,
    "lon_max": 13.7612,
}

# 1. Extract coordinates and predictions
lon = data["lon"].values
lat = data["lat"].values
z = data["value"].values

# 2. Define grid over bounding box (or data extent)
lon_grid = np.linspace(BBOX["lon_min"], BBOX["lon_max"], 200)
lat_grid = np.linspace(BBOX["lat_min"], BBOX["lat_max"], 200)
Lon, Lat = np.meshgrid(lon_grid, lat_grid)

# 3. Fit ordinary kriging
OK = OrdinaryKriging(
    lon, lat, z,
    variogram_model='spherical',
    verbose=False,
    enable_plotting=False
)

# 4. Compute kriging on the grid
Z_kriged, ss = OK.execute('grid', lon_grid, lat_grid)

# 5. Flatten grid into DataFrame
kriged_df = pd.DataFrame({
    "timestamp": data["timestamp"].max(),  # latest prediction timestamp
    "lat": Lat.ravel(),
    "lon": Lon.ravel(),
    "item_id": [None] * Lon.size,
    "value": Z_kriged.ravel(),
    "is_sensor": [False] * Lon.size
})

# 6. Add the original sensor points
sensors_df = data.rename(columns={"predictions": "value"}).copy()
sensors_df["is_sensor"] = True
final_df = pd.concat([
    sensors_df[["timestamp","lat","lon","item_id","value","is_sensor"]],
    kriged_df
], ignore_index=True)

final_df.head()


# In[7]:


print(final_df.columns)

final_df[~final_df['is_sensor']]['timestamp'].unique()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# 4. Plot heatmap
plt.figure(figsize=(8, 6))
plt.pcolormesh(Lon, Lat, Z_kriged, shading="auto")
plt.scatter(lon, lat, c=z, edgecolor="black", s=40)
plt.colorbar(label="Prediction")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Kriging Heatmap")
plt.show()

