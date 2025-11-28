# Ingest data infrastructure

This sets up a service that reads all the sensors daily from the site: https://archive.sensor.community/

Run the following:
1. `docker build -f Dockerfile.ingest -t chandlernick/sds011:latest .`
2. `docker push chandlernick/sds011-ingest:latest`
3. `kubectl apply -f data-pvc.yaml`
4. `kubectl apply -f ingest.yaml`
