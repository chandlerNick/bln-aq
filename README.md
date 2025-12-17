# BLN-AQ

## Description

BLN-AQ is a distributed **spatio-temporal modeling system** for estimating and forecasting air quality in Berlin, Germany, with a focus on **PM2.5 measurements from citizen-science sensors**.

The project combines:

- **Time-series forecasting** using foundation models  
- **Spatial interpolation** via geostatistical methods  

to generate **continuous air-quality estimates** over the Berlin metropolitan area.

Conceptually, this project builds on techniques explored in [BerlinWeatherTimeSeriesAnalysis](https://github.com/chandlerNick/BerlinWeatherTimeSeriesAnalysis), but extends them into the **spatial domain** and targets **particulate matter rather than meteorological variables**. Additionally, it uses more robust infrastructure and demonstrates ML systems architecture design and implementation.

---

## High-Level Architecture

The repository implements a distributed data pipeline composed of continuously running services and on-demand batch jobs:

Sensor Community -> Ingest -> Aggregate -> Forecast + Interpolate -> Frontend


---

## Directory Overview

### `prototyping/`
Contains experimental notebooks used to evaluate forecasting approaches and spatial interpolation behavior.

### `data/`
Stores historical weather data, intermediate artifacts, and produced forecasts.

### `data-engineering/sds011/`
Contains the production pipeline code responsible for ingestion, processing, forecasting, and visualization.

---

## `data-engineering/sds011` Pipeline Structure

This directory contains containerized services and batch-compute jobs orchestrated using **Kubernetes**.

### Components

### 1. `ingest`
Continuously downloads particulate matter data from `archive.sensor.community` and stores daily CSV dumps in persistent volume storage (PVC).

---

### 2. `aggregate`
Uses **Apache Spark** to:

- Clean and harmonize sensor time series  
- Resample measurements  
- Align timestamps  
- Produce **Parquet** datasets for downstream modeling  

This step is executed on demand to avoid idling expensive compute resources.

---

### 3. `predict-interpolate`

Performs two tightly coupled tasks:

#### Forecasting
Uses **Chronos2** in *in-context learning* mode to produce **3-day forecasts** of average PM2.5 for each sensor time series.

#### Spatial Interpolation
Uses **PyKrige** to krige the predicted pollution fields over a regular grid covering Berlin, producing continuous spatial estimates even where no sensors are physically present.

Outputs are written to Parquet for efficient reuse.

---

### 4. `front-end`

A **Streamlit** service that visualizes historical forecasts and interpolated pollution fields on an interactive Berlin map.

---

## Orchestration

The `ingest` and `front-end` services run continuously.

The `aggregate` and `predict-interpolate` jobs are executed via `orchestrate-frontend.sh` in order to prevent expensive resources (GPUs and Spark clusters) from idling.

This design cleanly separates:

- Real-time ingestion and presentation  
from  
- Batch-oriented compute-heavy modeling jobs

---

## Technology Stack

### Infrastructure
- Docker  
- Kubernetes  
- Persistent Volume Claims (PVCs)

### Data Engineering
- Apache Spark  
- Parquet  
- Python ETL pipelines

### Modeling and Statistics
- Chronos2 (time-series foundation model)  
- PyKrige (spatial interpolation)  
- NumPy, pandas, SciPy  

### Visualization
- Streamlit  
- Geospatial plotting and mapping libraries  

---

## Motivation

Berlin’s air-quality sensor network is **sparse, noisy, and heterogeneous**.  
This project bridges these deficiencies by combining:

- Foundation models for **temporal generalization**  
- Geostatistical techniques for **spatial inference**

to produce policy-relevant pollution maps from citizen science data.

---

## Deliverables

- Forecasted pollutant time series per sensor  
- Interpolated pollution maps over Berlin  
- A reproducible data and modeling pipeline  
- Presentation at `UT-Presentation.pdf`

---
