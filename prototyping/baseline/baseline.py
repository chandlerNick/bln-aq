#!/usr/bin/env python
# coding: utf-8

"""
Sliding-window evaluation for per-sensor ARIMA
Forecasts 3 days ahead, computes per-sensor and aggregated metrics, and times execution.
"""

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
import time

warnings.filterwarnings("ignore")

# ----------------------------
# Load data
# ----------------------------
df = pd.read_parquet("/storage/bln-aq/data/2024-citsci-pm2.5-daily.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Wide form: one column per sensor
df_wide = df.pivot(index="timestamp", columns="item_id", values="target").asfreq("D")

# Keep only sensors with >=10 daily points
sensor_counts = df_wide.notna().sum()
sensors_to_keep = sensor_counts[sensor_counts >= 10].index
df_wide = df_wide[sensors_to_keep]

# ----------------------------
# Parameters
# ----------------------------
WINDOW_DAYS = 60
FORECAST_DAYS = 3
ALPHA = 0.2  # for 10–90% quantiles

# Candidate ARIMA orders
ARIMA_CANDIDATES = [(1,0,1), (2,0,1), (1,1,1)]

# ----------------------------
# Prepare results containers
# ----------------------------
arima_preds = []
metrics_per_sensor = {}

# ----------------------------
# Sliding window evaluation with timing and batch ID
# ----------------------------
dates = df_wide.index
start_idx = WINDOW_DAYS
batch_id = 0

start_time = time.time()  # overall timing

for t_idx in range(start_idx, len(dates) - FORECAST_DAYS + 1):
    batch_id += 1
    train_window = df_wide.iloc[t_idx-WINDOW_DAYS:t_idx]
    forecast_dates = dates[t_idx:t_idx+FORECAST_DAYS]

    for sensor in df_wide.columns:
        ts = train_window[sensor].dropna()
        if len(ts) == 0:
            continue

        # ---------- ARIMA ----------
        best_fit = None
        best_aic = float("inf")
        for order in ARIMA_CANDIDATES:
            try:
                m = ARIMA(ts, order=order)
                f = m.fit()
                if f.aic < best_aic:
                    best_aic = f.aic
                    best_fit = f
            except:
                continue

        if best_fit is not None:
            forecast = best_fit.get_forecast(steps=FORECAST_DAYS)
            mean = forecast.predicted_mean
            conf = forecast.conf_int(alpha=ALPHA)
            p10 = conf.iloc[:,0]
            p90 = conf.iloc[:,1]

            for dt, m_val, low, high in zip(forecast_dates, mean, p10, p90):
                arima_preds.append({
                    "item_id": sensor,
                    "timestamp": dt,
                    "batch_id": batch_id,
                    "target_name": "arima",
                    "predictions": m_val,
                    "0.1": low,
                    "0.5": m_val,
                    "0.9": high
                })

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total sliding-window ARIMA predictions completed in {elapsed_time:.2f}s")  # 6126.09s

# ----------------------------
# Save predictions
# ----------------------------
df_arima = pd.DataFrame(arima_preds)
df_arima.to_csv("/storage/bln-aq/data/arima_daily_forecast_sliding.csv", index=False)
print("Saved ARIMA sliding forecasts -> /storage/bln-aq/data/arima_daily_forecast_sliding.csv")

