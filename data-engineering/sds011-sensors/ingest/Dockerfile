# Dockerfile
FROM python:3.11-slim

RUN pip install --no-cache-dir pandas requests

WORKDIR /app
COPY ingest_sds011.py .

# Default raw storage path (PVC will mount here)
VOLUME ["/data"]

CMD ["python", "ingest_sds011.py"]