# Aggregate data infrastructure

This infra allows the aggregation of the .csv files in the mini datalake via pyspark to a .parquet file for data analytics/ml

Note the data in the produced .parquet can be filtered via the .yaml (see commented section in `aggregate-job.yaml`).

To deploy (first time):
1. `docker build -t chandlernick/aggregate-sds011:latest -f Dockerfile.aggregate .`
2. `docker push chandlernick/aggregate-sds011:latest`

To run aggregation:
1. `kubectl apply -f sds011-aggregate-job.yaml`
2. `kubectl cp ...`  (filename: sds011.parquet)