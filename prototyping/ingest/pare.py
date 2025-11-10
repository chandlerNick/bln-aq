# Shrink the amount of data in the parquet (only bln sensors, hourly)

import polars as pl

# --- Config ---
INPUT_FILE = "../data/2024-citsci-pollutants.parquet"
OUTPUT_FILE = "../data/2024-citsci-pollutants-hourly.parquet"

BBOX = {
    "lat_min": 52.3383,
    "lat_max": 52.6755,
    "lon_min": 13.0884,
    "lon_max": 13.7612,
}

# --- Step 1: Read parquet ---
df = pl.read_parquet(INPUT_FILE)

# --- Step 2: Ensure timestamp is datetime ---
# If timestamp is string, convert to datetime
if df["timestamp"].dtype == pl.Utf8:
    df = df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S", strict=False)
    )


# --- Step 3: Round to nearest hour ---
df = df.with_columns(
    pl.col("timestamp").dt.round("1h").alias("timestamp_hour")
)

# --- Step 4: Filter Berlin bounding box ---
df = df.filter(
    (pl.col("lat").is_between(BBOX["lat_min"], BBOX["lat_max"])) &
    (pl.col("lon").is_between(BBOX["lon_min"], BBOX["lon_max"]))
)

# --- Step 5: Drop/rename cols ---
df = df.drop("timestamp")
df = df.drop("sensor_id")
df = df.drop("P1")  # Remove PM10 - PM2.5 is more interesting for public health.
df = df.rename({"P2":"PM2_5"})

print(df.head())

# --- Step 6: Save to parquet ---
df.write_parquet(OUTPUT_FILE)

print(f"Saved filtered and rounded data to {OUTPUT_FILE}")