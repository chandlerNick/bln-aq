#!/usr/bin/env python3
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp, from_unixtime, to_timestamp, lit
from pyspark.sql.types import TimestampType

# ------------------------------
# CONFIGURATION FROM ENVIRONMENT
# ------------------------------
RAW_DIR = os.getenv("RAW_DIR", "/data/raw")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/data/parquet")

sensor_env = os.getenv("SENSOR_IDS")  # comma-separated, e.g. "828,1376"
SENSOR_IDS = [int(x) for x in sensor_env.split(",")] if sensor_env else None

columns_env = os.getenv("COLUMNS")  # e.g. "timestamp,lat,lon,P2"
if columns_env:
    COLUMNS = [c.strip() for c in columns_env.split(",") if c.strip()]
else:
    COLUMNS = None  # None -> keep all columns

date_range_env = os.getenv("DATE_RANGE")  # "2024-01-01,2024-01-31"
DATE_RANGE = tuple(date_range_env.split(",")) if date_range_env else None

# ------------------------------
# INIT SPARK
# ------------------------------
spark = SparkSession.builder.appName("SDS011 Quantize").getOrCreate()

# ------------------------------
# COLLECT CSV FILES
# ------------------------------
if not os.path.isdir(RAW_DIR):
    print(f"ERROR: RAW_DIR does not exist or is not a directory: {RAW_DIR}", file=sys.stderr)
    sys.exit(2)

all_csvs = [os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.lower().endswith(".csv")]
if not all_csvs:
    print(f"ERROR: No CSV files found in {RAW_DIR}", file=sys.stderr)
    sys.exit(3)

# ------------------------------
# READ CSVs (semicolon separated)
# ------------------------------
# Let Spark infer schema lightly but parse timestamp strings later explicitly.
df = spark.read.option("header", True).option("sep", ";").csv(all_csvs)

# ------------------------------
# Ensure timestamp column exists
# ------------------------------
if "timestamp" not in df.columns:
    print("ERROR: 'timestamp' column not found in input CSVs.", file=sys.stderr)
    sys.exit(4)

# ------------------------------
# Parse timestamp into real TimestampType
# ------------------------------
# Accept ISO-like timestamps; fallback to nulls if unparsable.
df = df.withColumn("timestamp_ts", to_timestamp(col("timestamp")))

# ------------------------------
# OPTIONAL: date range filter (compare against timestamp)
# ------------------------------
if DATE_RANGE is not None:
    start_date, end_date = DATE_RANGE
    # Convert to timestamp for comparison (midnight local -> treat as UTC-like input)
    df = df.filter((col("timestamp_ts") >= to_timestamp(lit(start_date))) &
                   (col("timestamp_ts") <= to_timestamp(lit(end_date))))

# ------------------------------
# OPTIONAL: sensor filter
# ------------------------------
if SENSOR_IDS is not None:
    if "sensor_id" not in df.columns:
        print("ERROR: 'sensor_id' column not present but SENSOR_IDS was provided.", file=sys.stderr)
        sys.exit(5)
    df = df.filter(col("sensor_id").cast("int").isin(SENSOR_IDS))

# ------------------------------
# ROUND TIMESTAMP TO NEAREST MINUTE (server-side, no UDF)
# Approach:
#   unix_ts = unix_timestamp(timestamp_ts)           -> seconds since epoch
#   rounded_secs = ((unix_ts + 30) / 60).cast('long') * 60
#   timestamp_rounded = from_unixtime(rounded_secs).cast(TimestampType())
# This rounds to nearest minute (seconds >=30 round up).
# ------------------------------
unix_ts = unix_timestamp(col("timestamp_ts"))
rounded_min_secs = ( (unix_ts + lit(30)) / lit(60) ).cast("long") * lit(60)
df = df.withColumn("timestamp_rounded", from_unixtime(rounded_min_secs).cast(TimestampType()))

# ------------------------------
# SELECT COLUMNS: keep user-specified columns or all, then append rounded timestamp
# ------------------------------
if COLUMNS is None:
    # keep all original columns (preserve order) plus timestamp_rounded
    select_cols = [c for c in df.columns if c != "timestamp_ts"]  # keep original timestamp string if present
else:
    # validate requested columns exist
    missing = [c for c in COLUMNS if c not in df.columns and c != "timestamp_rounded"]
    if missing:
        print(f"ERROR: Requested columns not found in data: {missing}", file=sys.stderr)
        sys.exit(6)
    select_cols = COLUMNS.copy()

# ensure we don't duplicate timestamp_rounded
if "timestamp_rounded" not in select_cols:
    select_cols.append("timestamp_rounded")

df_out = df.select(*select_cols)

# ------------------------------
# WRITE PARQUET
# ------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "sds011.parquet")

# Use overwrite mode as default
df_out.write.mode("overwrite").parquet(out_path)

spark.stop()
print(f"Parquet written to {out_path}")
