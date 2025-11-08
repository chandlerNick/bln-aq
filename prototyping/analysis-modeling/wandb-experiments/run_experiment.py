# run_experiment.py
import os
import torch, wandb, yaml
from data_utils import load_and_prepare_data, prepare_tensors
from model import SpatioTemporalGP
from train import train_model
from evaluate import evaluate_model
import gpytorch
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

def make_inducing_points(strategy, grid_size, train_X_np, scaler_X, device, time_median=None):
    if strategy == "kmeans":
        # cluster on (lat, lon, minutes_since_start)
        kmeans = KMeans(n_clusters=grid_size, random_state=42).fit(train_X_np)
        pts = kmeans.cluster_centers_
    else:  # grid
        lat_min, lat_max = train_X_np[:,0].min(), train_X_np[:,0].max()
        lon_min, lon_max = train_X_np[:,1].min(), train_X_np[:,1].max()
        lat_grid = np.linspace(lat_min, lat_max, int(np.sqrt(grid_size)))
        lon_grid = np.linspace(lon_min, lon_max, int(np.sqrt(grid_size)))
        lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid)
        time_val = time_median if time_median is not None else np.median(train_X_np[:,2])
        pts = np.column_stack([lat_mesh.ravel(), lon_mesh.ravel(), np.full(lat_mesh.size, time_val)])
    pts_scaled = scaler_X.transform(pts)
    inducing = torch.tensor(pts_scaled, dtype=torch.float32).to(device)
    return inducing

def main():
    with open("config/default.yaml") as f:
        defaults = yaml.safe_load(f)

    # use wandb config (sweep will override)
    wandb.init(project="berlin-air-gpr", config=defaults)
    cfg = wandb.config

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    train_df, test_df = load_and_prepare_data(
        cfg['data_path'], cfg['lookback_days'], cfg['temp_horizon_hours'], cfg['spatial_test_frac']
    )

    X_train_t, y_train_t, X_test_t, y_test_t, scaler_X, scaler_y = prepare_tensors(train_df, test_df, device=device)

    # convert to numpy for kmeans/grid logic
    X_train_np = X_train_t.cpu().numpy()
    # choose inducing strategy
    inducing = make_inducing_points(cfg.get('inducing_strategy', 'grid'),
                                    cfg.get('grid_size', 25),
                                    X_train_np, scaler_X, device,
                                    time_median=np.median(X_train_np[:,2]))

    # instantiate model with kernel_cfg from wandb config
    kernel_cfg = {
        "spatial_kernel": cfg.get('spatial_kernel', 'Matern'),
        "temporal_kernel": cfg.get('temporal_kernel', 'Periodic'),
        "matern_nu": cfg.get('matern_nu', 1.5),
        "period_length": cfg.get('period_length', 24.0),
        "combine": cfg.get('kernel_combine', 'sum'),
        "learn_inducing": cfg.get('learn_inducing', True)
    }

    model = SpatioTemporalGP(inducing, kernel_cfg=kernel_cfg).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=float(cfg['learning_rate']))

    num_data = X_train_t.size(0)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)

    model, likelihood = train_model(
        model, likelihood, optimizer, mll, X_train_t, y_train_t, X_test_t, y_test_t, dict(cfg), scaler_y, device=device
    )

    # evaluate
    rmse, r2 = evaluate_model(model, likelihood, X_test_t, y_test_t, scaler_y)
    wandb.log({"RMSE": rmse, "R2": r2})
    print(f"Final RMSE: {rmse:.3f}, R²: {r2:.3f}")

    wandb.finish()

if __name__ == "__main__":
    main()
