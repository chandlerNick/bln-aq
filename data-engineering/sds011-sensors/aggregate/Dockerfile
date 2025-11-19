# Start from Apache Spark image (Python and Java included)
FROM apache/spark:3.5.0-python3

# Install Python dependencies only
USER root
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install pandas pyarrow pyspark

# Set working directory
WORKDIR /app
COPY aggregate_sds011.py .

# PVC mount point
VOLUME ["/data"]

# Run aggregation script with the system Python
CMD ["python3", "aggregate_sds011.py"]