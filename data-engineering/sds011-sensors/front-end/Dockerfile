FROM python:3.12-slim

RUN pip install --no-cache-dir \
    streamlit \
    pandas \
    numpy \
    geopandas \
    matplotlib \
    pyproj \
    shapely

WORKDIR /app

COPY streamlit_app.py .
COPY bezirke.geojson .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
