import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs

# ----------------------------
# Paths
# ----------------------------
DATA_DIR = Path("/data/parquet")   # mounted PV
GEO_FILE = Path("/app/bezirke.geojson")  # baked into container

# REMOVE BEFORE FLIGHT
DATA_DIR = Path("../tester-data")
GEO_FILE = Path("./bezirke.geojson")

# Load bezirke polygons once
bezirke = gpd.read_file(GEO_FILE).to_crs(epsg=4326)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Berlin PM2.5 Forecast", layout="wide")
st.title("Berlin PM2.5 Forecast (Kriged)")

# Dynamically list available dates
dates = sorted([f.stem.replace("predicted_data_", "") for f in DATA_DIR.glob("predicted_data_*.parquet")])
if not dates:
    st.error("No forecast data found in /data/parquet!")
    st.stop()

selected_date = st.selectbox("Select forecast date", dates)

# ----------------------------
# Load data for selected date
# ----------------------------
pred_file = DATA_DIR / f"predicted_data_{selected_date}.parquet"
grid_file = DATA_DIR / f"Z_kriged_{selected_date}.npy"

data = pd.read_parquet(pred_file)
Z_kriged = np.load(grid_file)

# Sensor coordinates and values
lon = data["lon"].astype(float).values
lat = data["lat"].astype(float).values
values = data["value"].astype(float).values

# ----------------------------
# Prepare grid
# ----------------------------
# Assuming fixed BBOX and grid shape
BBOX = {
    "lat_min": 52.3383,
    "lat_max": 52.6755,
    "lon_min": 13.0884,
    "lon_max": 13.7612,
}
buffer = 0.01
ny, nx = Z_kriged.shape
lon_grid = np.linspace(BBOX['lon_min'] - buffer, BBOX['lon_max'] + buffer, nx)
lat_grid = np.linspace(BBOX['lat_min'] - buffer, BBOX['lat_max'] + buffer, ny)
Lon, Lat = np.meshgrid(lon_grid, lat_grid)

# ----------------------------
# Plotting
# ----------------------------
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
im = ax.pcolormesh(Lon, Lat, Z_kriged, shading='auto', cmap='viridis', alpha=0.7, transform=ccrs.PlateCarree())
ax.scatter(lon, lat, c=values, edgecolor='black', s=40, transform=ccrs.PlateCarree(), cmap='viridis')

for geom in bezirke.geometry:
    ax.add_geometries([geom], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=1)

# Colorbar (auto-scaled)
vmin, vmax = np.nanmin(Z_kriged), np.nanmax(Z_kriged)
sm = mpl.cm.ScalarMappable(cmap='viridis', norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.04)
cbar.set_label("PM2.5 (µg/m³)")

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"PM2.5 Kriging Heatmap - {selected_date}")

st.pyplot(fig)
