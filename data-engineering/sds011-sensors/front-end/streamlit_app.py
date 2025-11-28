import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import pydeck as pdk
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import io

# ----------------------------
# Paths
# ----------------------------
DATA_DIR = Path(os.environ.get("PARQUET_DIR", "/data/parquet"))
#DATA_DIR = Path("../tester-data")  # adjust for production

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
sensor_df = data[data['is_sensor']]
kriged_df = data[~data['is_sensor']]


# ----------------------------
# Color mapping using viridis
# ----------------------------
cmap = plt.colormaps["viridis"]
norm = mcolors.Normalize(vmin=data['value'].min(), vmax=data['value'].max())

kriged_colors = (cmap(norm(kriged_df['value'].values))[:, :3] * 255).astype(int)
sensor_colors = (cmap(norm(sensor_df['value'].values))[:, :3] * 255).astype(int)


# Add a column with color to each DataFrame
kriged_df = kriged_df.copy()
kriged_df["color"] = kriged_colors.tolist()

sensor_df = sensor_df.copy()
sensor_df["color"] = sensor_colors.tolist()


# ----------------------------
# Pydeck layers
# ----------------------------
layers = [
    pdk.Layer(
        "ScatterplotLayer",
        data=kriged_df,
        get_position='[lon, lat]',
        get_fill_color='color',
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
        get_fill_color='color',
        get_radius=100,
        pickable=True,
        auto_highlight=True,
        radius_min_pixels=5,
        radius_max_pixels=12,
        tooltip=True
    )
]

# ----------------------------
# Pydeck map
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

# ----------------------------
# Add color legend
# ----------------------------
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=ax, orientation='horizontal')
cbar.set_label("PM2.5 [µg/m³]")

buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches='tight')
buf.seek(0)
st.image(buf)
