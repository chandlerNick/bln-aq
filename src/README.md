# Project Overview

Having explored the space a bit, I want to deliver two things:
1. A demonstrator that goes from API to Grafana Dashboard
2. An experimental analysis showing that chronos2 + Ordinary Kriging (OK) > (ARIMA + OK), (Chronos + LinInterp), (ARIMA + LinInterp). In the process, we get metrics for chronos2 + OK


The demo has the following structure (hits a bunch of key tech for data engineering):
1. API
2. Kafka
3. Spark Structured Streaming <- data clean and aggregate
4. Delta Feature Store
5. Pred. Job (Chronos2)
6. Interp. Job (OK)
7. PostGIS
8. Grafana Dashboard

The experimental analysis does the following:
1. Do rolling window TS preds over one year
2. Do spatial interpolation
3. Wrap 1, 2 in 2D-K-Fold Cross Validation (spatially CV'd and temporally CV'd)
4. Report table of metrics for each scenario. RMSE for example.
