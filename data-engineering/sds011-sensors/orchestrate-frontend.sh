#!/bin/bash
set -euo pipefail

NAMESPACE="nich1834"  # change if needed
AGGREGATE_JOB="sds011-aggregate"
PREDICT_JOB="sds011-predict"
FRONTEND_DEPLOY="pm25-streamlit"

echo "=== Starting SDS011 pipeline ==="

# Apply the aggregate job
echo "Applying aggregate job..."
kubectl apply -n $NAMESPACE -f aggregate/aggregate-job.yaml

echo "Waiting for aggregate job to complete..."
kubectl wait --for=condition=complete job/$AGGREGATE_JOB -n $NAMESPACE --timeout=30m
echo "Aggregate job completed"

# Apply the prediction job
echo "Applying prediction job..."
kubectl apply -n $NAMESPACE -f predict-interpolate/pred-interp.yaml

echo "Waiting for prediction job to complete..."
kubectl wait --for=condition=complete job/$PREDICT_JOB -n $NAMESPACE --timeout=60m
echo "Prediction job completed"

# Restart the frontend deployment
echo "Restarting frontend deployment..."
kubectl rollout restart deployment/$FRONTEND_DEPLOY -n $NAMESPACE
echo "Frontend deployment restarted"

echo "=== SDS011 pipeline finished successfully ==="
