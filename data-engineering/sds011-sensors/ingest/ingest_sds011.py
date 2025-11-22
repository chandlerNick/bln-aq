#!/usr/bin/env python3
import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

RAW_DIR = "/data/raw"
MAX_WORKERS = 16
TIMEOUT = 120

SENSOR_IDS = [
    828,1376,1412,1847,2055,2107,2119,2123,2211,2618,2724,3036,3355,3491,3563,3755,
    3943,3979,4915,4961,5026,5040,5355,5706,5710,6119,6209,6438,6537,6825,6928,7051,
    7158,7203,7263,8092,8654,8675,9304,9310,9392,9409,9767,9809,10162,11957,12171,12603,
    12762,13090,13197,13366,13368,13588,13733,13834,14681,15293,15317,15536,15563,15843,
    16312,16866,17686,18376,18473,18796,18879,18983,19095,19645,19727,20478,20826,20861,
    20972,21334,22184,22240,22826,23322,23472,25758,25899,26261,26365,26696,26912,26951,
    27293,27517,27812,28245,28359,28489,29024,29150,29380,31204,31298,31450,32501,32974,
    33698,33796,35795,35807,36091,36899,37449,37517,37519,38174,40765,40807,41261,41943,
    42226,43337,43626,43640,44120,44310,44720,45762,47575,47966,50066,50221,50549,51034,
    51354,51616,53811,53931,56680,56927,57233,60516,64413,65667,68306,68686,68976,69008,
    70078,70334,70526,70607,70903,71083,71462,72065,72203,72235,72520,73881,74789,77178,
    78143,78412,78796,80243,80322,80508,80977,81487,81508,83560,83562,84042,84095,84347,
    84379,85065,85498,85556,85676,85785,86868,87048,87165,87630,87730,87763,87879,88152,
    88791,88808,89117,89586,89894,90248
]


def download_sensor_csv(date_str: str, sid: int) -> str:
    """Download one sensor CSV and save to RAW_DIR if available."""
    file_name = f"{date_str}_sds011_sensor_{sid}.csv"
    url = f"https://archive.sensor.community/{date_str}/{file_name}"
    out_path = os.path.join(RAW_DIR, file_name)

    if os.path.exists(out_path):
        return f"Skipped existing: {file_name}"

    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 200:
            df = pd.read_csv(StringIO(r.text))
            df.to_csv(out_path, index=False)
            return f"Saved {file_name} ({len(df)} rows)"
        elif r.status_code == 404:
            return f"Missing: {file_name}"
        else:
            return f"HTTP {r.status_code}: {file_name}"
    except Exception as e:
        return f"Error for {file_name}: {e}"


def fetch_for_date(target_date: datetime):
    date_str = target_date.strftime("%Y-%m-%d")
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Fetching SDS011 data for {date_str}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_sensor_csv, date_str, sid): sid for sid in SENSOR_IDS}
        for i, future in enumerate(as_completed(futures), 1):
            msg = future.result()
            print(f"[{i}/{len(SENSOR_IDS)}] {msg}")


def main():
    target_date = datetime.now(timezone.utc) - timedelta(days=1)
    fetch_for_date(target_date)


if __name__ == "__main__":
    main()
