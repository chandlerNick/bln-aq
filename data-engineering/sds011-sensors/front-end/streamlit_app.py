import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import pydeck as pdk
import os

# ----------------------------
# Paths
# ----------------------------
DATA_DIR = Path(os.environ.get("PARQUET_DIR", "/data/parquet"))
DATA_DIR = Path("../tester-data")  # adjust for production

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Berlin PM2.5 Forecast", layout="wide")
st.title("Berlin PM2.5 Kriging Forecast (Interactive)")

# List forecast dates
dates = sorted([f.stem.replace("predicted_plus_kriged_", "") for f in DATA_DIR.glob("predicted_plus_kriged_*.parquet")])
if not dates:
    st.error(f"No forecast data found in {DATA_DIR}!")
    st.stop()

selected_date = st.selectbox("Select forecast date", dates)

# ----------------------------
# Load forecast data
# ----------------------------
pred_file = DATA_DIR / f"predicted_plus_kriged_{selected_date}.parquet"
if not pred_file.exists():
    st.error(f"Forecast file missing for {selected_date}")
    st.stop()

data = pd.read_parquet(pred_file)

# ----------------------------
# Prepare pydeck layers
# ----------------------------

# Sensor points
sensor_df = data[data['is_sensor']]
kriged_df = data[~data['is_sensor']]

# Function to normalize values to 0-255 for color mapping
def normalize_color(vals, cmap_min=None, cmap_max=None):
    if cmap_min is None: cmap_min = np.nanmin(vals)
    if cmap_max is None: cmap_max = np.nanmax(vals)
    norm = ((vals - cmap_min) / (cmap_max - cmap_min) * 255).astype(int)
    return norm

# Compute color values (R,G,B)
cmap_min, cmap_max = data['value'].min(), data['value'].max()
kriged_colors = np.stack([normalize_color(kriged_df['value'], cmap_min, cmap_max),
                          np.zeros(len(kriged_df)),
                          255 - normalize_color(kriged_df['value'], cmap_min, cmap_max)], axis=1)

sensor_colors = np.stack([np.zeros(len(sensor_df)), 
                          255 - normalize_color(sensor_df['value'], cmap_min, cmap_max), 
                          normalize_color(sensor_df['value'], cmap_min, cmap_max)], axis=1)

# Pydeck layers
layers = [
    pdk.Layer(
        "ScatterplotLayer",
        data=kriged_df,
        get_position='[lon, lat]',
        get_fill_color=kriged_colors.tolist(),
        get_radius=50,
        pickable=True,
        auto_highlight=True,
        radius_min_pixels=2,
        radius_max_pixels=10,
        tooltip=True
    ),
    pdk.Layer(
        "ScatterplotLayer",
        data=sensor_df,
        get_position='[lon, lat]',
        get_fill_color=sensor_colors.tolist(),
        get_radius=100,
        pickable=True,
        auto_highlight=True,
        radius_min_pixels=5,
        radius_max_pixels=12,
        tooltip=True
    )
]

# ----------------------------
# Pydeck Map
# ----------------------------
view_state = pdk.ViewState(
    longitude=13.405,
    latitude=52.52,
    zoom=11,
    pitch=0
)

r = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    tooltip={
        "html": "<b>Value:</b> {value} µg/m³<br><b>0.1:</b> {0.1} µg/m³<br><b>0.9:</b> {0.9} µg/m³",
        "style": {"color": "white"}
    },
    map_style='light'
)

st.pydeck_chart(r)
