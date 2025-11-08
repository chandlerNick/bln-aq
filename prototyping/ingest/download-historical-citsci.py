import requests
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================
# CONFIGURATION
# ==============================
DATA_DIR = Path("/mnt/data/berlin-data")
DATA_DIR.mkdir(exist_ok=True)

SENSOR_COMMUNITY_URL = "https://data.sensor.community/static/v2/data.json"
ARCHIVE_BASE = "https://archive.sensor.community"
SENSOR_TYPE = "sds011"
YEAR = 2024

MAX_WORKERS = 10  # number of concurrent threads

# Berlin bounding box
BBOX = {
    "lat_min": 52.3383,
    "lat_max": 52.6755,
    "lon_min": 13.0884,
    "lon_max": 13.7612,
}

def in_berlin(lat, lon):
    return BBOX["lat_min"] <= lat <= BBOX["lat_max"] and BBOX["lon_min"] <= lon <= BBOX["lon_max"]

# ==============================
# GET SDS011 SENSOR IDS IN BERLIN
# ==============================
print("Fetching SDS011 sensor metadata...")
resp = requests.get(SENSOR_COMMUNITY_URL)
resp.raise_for_status()
data = resp.json()

sensor_ids = set()
for entry in data:
    try:
        if entry["sensor"]["sensor_type"]["name"].lower() == "sds011":
            lat = float(entry["location"]["latitude"])
            lon = float(entry["location"]["longitude"])
            if in_berlin(lat, lon):
                sensor_ids.add(entry["sensor"]["id"])
    
    except (KeyError, ValueError):
        continue

print(f"Found {len(sensor_ids)} SDS011 sensors in Berlin.")

# ==============================
# FUNCTION TO DOWNLOAD + FILTER CSV
# ==============================
def download_and_filter(sensor_id, day):
    ds = day.strftime("%Y-%m-%d")
    out_file = DATA_DIR / f"{ds}_sensor_{sensor_id}.csv"

    # Skip if already downloaded
    if out_file.exists():
        print(f"Skipping existing {out_file}")
        return

    url = f"{ARCHIVE_BASE}/{ds}/{ds}_{SENSOR_TYPE}_sensor_{sensor_id}.csv"
    try:
        df = pd.read_csv(url, sep=";")
        if not df.empty:
            df_filtered = df[["timestamp", "P1", "P2", "lat", "lon"]].copy()
            df_filtered["sensor_id"] = sensor_id
            df_filtered.to_csv(out_file, index=False)
            print(f"Saved {out_file} ({len(df_filtered)} rows)")
    except Exception:
        print(f"Missing or failed CSV for sensor {sensor_id} on {ds}")

# ==============================
# GENERATE TASKS
# ==============================
start_date = date(YEAR, 1, 1)
end_date = date(YEAR, 12, 31)

tasks = [(sensor_id, start_date + timedelta(days=i))
         for sensor_id in sensor_ids
         for i in range((end_date - start_date).days + 1)]

print(f"📅 {len(tasks)} total sensor-day downloads")

# ==============================
# MULTITHREADED DOWNLOAD
# ==============================
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(download_and_filter, sid, day): (sid, day) for sid, day in tasks}

    completed = 0
    for future in as_completed(futures):
        completed += 1
        if completed % 100 == 0:
            print(f"Progress: {completed}/{len(futures)} ({completed/len(futures)*100:.2f}%)")

print("All downloads complete.")

