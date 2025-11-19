FROM python:3.12-slim

# Minimal system dependencies for compiling some Python packages
RUN apt-get update && apt-get install -y \
        build-essential \
        libssl-dev \
        libffi-dev \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY predict.py .

# Install Python dependencies
RUN pip install --no-cache-dir \
        pandas \
        pyarrow \
        chronos-forecasting>=2.0 \
        pykrige

CMD ["python", "predict.py"]
