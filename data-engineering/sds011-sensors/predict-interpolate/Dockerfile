FROM python:3.12-slim

# Minimal system dependencies for compiling Python packages and geospatial libs
RUN apt-get update && apt-get install -y \
        build-essential \
        libssl-dev \
        libffi-dev \
        libgomp1 \
        gdal-bin \
        libgdal-dev \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY predict.py .

# Install Python dependencies
RUN pip install --no-cache-dir \
        pandas \
        pyarrow \
        chronos-forecasting>=2.0 \
        pykrige \
        geopandas \
        pyproj \
        numpy

# Optional: set environment variable for parquet directory
ENV PARQUET_DIR=/data/parquet

CMD ["python", "predict.py"]
