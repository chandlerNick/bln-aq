# Urban Tech Core Project

This project concerns itself with taking citizen science sensor data, aggregating it, performing zero-shot time series forecasting, kriging to spatially interpolate the forecasts, and visualizing the results in a one-day-ahead map of Berlin.

The technologies used are:
- Python
- Docker
- Kubernetes
- Apache Spark
- Chronos2
- Kriging

To run the deployment:
1. Ensure all docker containers in the child directories of `sds011-sensors` are built.
2. Ensure `sds011-sensors/ingest` is running and has collected some data.
3. Run `./orchestrate-frontend.sh` to run the aggregate, predict-interpolate, and deploy-frontend jobs.

Access the streamlit frontend at: http://pm25.project.ris.bht-berlin.de/ (from within the BHT Network).
