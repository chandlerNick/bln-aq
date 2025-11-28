#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import geopandas as gpd
from pathlib import Path

# ----------------------------
# Paths
# ----------------------------
input_file = Path("../predicted_data.parquet")
BBOX = {
    "lat_min": 52.3383,
    "lat_max": 52.6755,
    "lon_min": 13.0884,
    "lon_max": 13.7612,
}

# ----------------------------
# Load cleaned data
# ----------------------------
data = pd.read_parquet(input_file)

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

# ----------------------------
# Load Berlin Bezirke GeoJSON
# ----------------------------
bezirke = gpd.read_file("bez.geojson").to_crs(epsg=4326)

# ----------------------------
# Plot
# ----------------------------
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
im = ax.pcolormesh(Lon, Lat, Z_kriged, shading='auto', cmap='viridis', alpha=0.7, transform=ccrs.PlateCarree())
ax.scatter(lon, lat, c=z, edgecolor='black', s=40, transform=ccrs.PlateCarree(), cmap='viridis')

for geom in bezirke.geometry:
    ax.add_geometries([geom], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=1)

sm = mpl.cm.ScalarMappable(cmap='viridis', norm=mpl.colors.Normalize(vmin=z.min(), vmax=z.max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.04)
cbar.set_label("PM2.5")

ax.set_title("Kriging Heatmap with Berlin Bezirke Overlay")
plt.savefig("fig.pdf")
