#!/usr/bin/env python
# coding: utf-8

# ## Spatiotemporal Forecasting + Interpolation w/ GPR
# 
# Take everything seen so far, and predict P1 and P2 on a regular grid over Berlin at a future timestamp using a spatiotemporal Gaussian Process.

# In[11]:


import pandas as pd
import numpy as np
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


BBOX = {
    "lat_min": 52.3383,
    "lat_max": 52.6755,
    "lon_min": 13.0884,
    "lon_max": 13.7612,
}


# ### Data ingest

# In[12]:


# ----------------------------
# Hyperparameters
# ----------------------------
LOOKBACK_DAYS = 21         # rolling lookback window in days
TEMP_HORIZON_HOURS = 1     # how far ahead to predict
SPATIAL_TEST_FRAC = 0.2    # fraction of locations held out for spatial test

# ----------------------------
# Read data
# ----------------------------
df = pd.read_parquet("../data/2024-citsci-pollutants-hourly.parquet")

# Ensure datetime
df['timestamp_hour'] = pd.to_datetime(df['timestamp_hour'])

# Aggregate PM2.5 per sensor/location + hour
df = df.groupby(['lat', 'lon', 'timestamp_hour'], as_index=False)['PM2_5'].mean()

# ----------------------------
# Rolling lookback
# ----------------------------
latest_time = df['timestamp_hour'].max()
start_time = latest_time - pd.Timedelta(days=LOOKBACK_DAYS)
df_recent = df[df['timestamp_hour'] >= start_time].copy()

# ----------------------------
# Spatio-temporal train/test split
# ----------------------------

# Select spatial test locations
unique_locations = df_recent[['lat', 'lon']].drop_duplicates()
num_test_locations = int(len(unique_locations) * SPATIAL_TEST_FRAC)
test_locations = unique_locations.sample(n=num_test_locations, random_state=42)

# Temporal split
train_end_time = latest_time - pd.Timedelta(hours=TEMP_HORIZON_HOURS)

# Training set: before temporal horizon AND not in spatial test locations
train_df = df_recent[
    (df_recent['timestamp_hour'] <= train_end_time) &
    (~df_recent.set_index(['lat', 'lon']).index.isin(test_locations.set_index(['lat', 'lon']).index))
].copy()

# Test set: either in spatial test locations OR after temporal horizon
test_df = df_recent[
    (df_recent['timestamp_hour'] > train_end_time) |
    (df_recent.set_index(['lat', 'lon']).index.isin(test_locations.set_index(['lat', 'lon']).index))
].copy()

# ----------------------------
# Convert time to minutes since start (relative to training start)
# ----------------------------
train_start_time = train_df['timestamp_hour'].min()
train_df['minutes_since_start'] = (train_df['timestamp_hour'] - train_start_time).dt.total_seconds() / 60
test_df['minutes_since_start'] = (test_df['timestamp_hour'] - train_start_time).dt.total_seconds() / 60

# ----------------------------
# Features & targets
# ----------------------------
X_train = train_df[['lat','lon','minutes_since_start']].values
y_train = train_df['PM2_5'].values
X_test = test_df[['lat','lon','minutes_since_start']].values
y_test = test_df['PM2_5'].values

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

# ----------------------------
# Device
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------
# Scale features and targets
# ----------------------------
scaler_X = StandardScaler().fit(X_train)
X_train_scaled = torch.tensor(scaler_X.transform(X_train), dtype=torch.float32).to(device)
X_test_scaled = torch.tensor(scaler_X.transform(X_test), dtype=torch.float32).to(device)

scaler_y = StandardScaler().fit(y_train.reshape(-1,1))
y_train_scaled = torch.tensor(scaler_y.transform(y_train.reshape(-1,1)), dtype=torch.float32).squeeze().to(device)
y_test_scaled = torch.tensor(scaler_y.transform(y_test.reshape(-1,1)), dtype=torch.float32).squeeze().to(device)


# In[ ]:


class SpatioTemporalGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, ard_dims=(2,1)):
        """
        Args:
            inducing_points: torch.Tensor of shape (M, D), subset of training points for variational GP
            ard_dims: tuple (num_spatial_dims, num_temporal_dims)
        """
        # Variational distribution
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        # ------------------------
        # Mean module
        # ------------------------
        self.mean_module = gpytorch.means.ConstantMean()

        # ------------------------
        # Kernel: Separate spatial and temporal kernels with ARD
        # ------------------------
        spatial_kernel = gpytorch.kernels.RBFKernel(
            ard_num_dims=ard_dims[0]
        )
        temporal_kernel = gpytorch.kernels.RBFKernel(
            ard_num_dims=ard_dims[1]
        )

        # Combine kernels via product for spatio-temporal interaction
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.ProductKernel(spatial_kernel, temporal_kernel)
        )

    def forward(self, x):
        """
        Forward pass.
        x: torch.Tensor of shape (N, 3) with columns [lat, lon, minutes_since_start]
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ### Instantiation

# In[ ]:


# ----------------------------
# Inducing points (subset of training data)
# ----------------------------
num_inducing = min(500, X_train_scaled.shape[0])  # adjust depending on GPU memory
inducing_points = X_train_scaled[torch.randperm(X_train_scaled.size(0))[:num_inducing]]

# ----------------------------
# Model + Likelihood
# ----------------------------
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model = SpatioTemporalGP(inducing_points).to(device)

# ----------------------------
# Training hyperparameters
# ----------------------------
training_iterations = 300
learning_rate = 0.01

# ----------------------------
# Optimizer
# ----------------------------
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()}
], lr=learning_rate)

# ----------------------------
# Loss (ELBO)
# ----------------------------
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_train_scaled.size(0))


# ### Train Loop

# In[ ]:


# ----------------------------
# Training loop
# ----------------------------
model.train()
likelihood.train()

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(X_train_scaled)
    loss = -mll(output, y_train_scaled)
    loss.backward()
    if (i+1) % 50 == 0:
        print(f"Iter {i+1}/{training_iterations} - Loss: {loss.item():.4f}")
    optimizer.step()


# In[ ]:


# Create a grid over Berlin (lat/lon + target future time)
lat_grid = np.linspace(52.3383, 52.6755, 50)
lon_grid = np.linspace(13.0884, 13.7612, 50)
minutes_future = X_tensor_scaled[:, 2].max().item() + 60  # 1h ahead

grid = np.array([[lat, lon, minutes_future] for lat in lat_grid for lon in lon_grid], dtype=np.float32)
grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)

model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred = likelihood(model(grid_tensor))

    # Move to CPU and reshape for scaler
    y_pred_mean_scaled = pred.mean.cpu().numpy().reshape(-1, 1)
    y_pred = y_scaler.inverse_transform(y_pred_mean_scaled).flatten()

    # Standard deviation: scale by scaler.scale_
    y_std = (pred.stddev.cpu().numpy() * y_scaler.scale_).flatten()


# In[ ]:


# Reshape predictions to grid shape
lat_len = len(lat_grid)
lon_len = len(lon_grid)

y_pred_grid = y_pred.reshape(lat_len, lon_len)
y_std_grid = y_std.reshape(lat_len, lon_len)

# Create meshgrid for plotting
Lon, Lat = np.meshgrid(lon_grid, lat_grid)

plt.figure(figsize=(10, 8))

# Plot mean prediction as a heatmap
plt.pcolormesh(Lon, Lat, y_pred_grid, shading='auto', cmap='viridis')
plt.colorbar(label='PM2.5 (μg/m³)')
plt.scatter(df['lon'], df['lat'], c='red', s=10, alpha=0.5, label='Sensor locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'GP Prediction of PM2.5 1h Ahead over Berlin')
plt.legend()
plt.show()

# Optional: plot uncertainty
plt.figure(figsize=(10, 8))
plt.pcolormesh(Lon, Lat, y_std_grid, shading='auto', cmap='inferno')
plt.colorbar(label='Predicted Stddev')
plt.scatter(df['lon'], df['lat'], c='white', s=10, alpha=0.5, label='Sensor locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('GP Prediction Uncertainty')
plt.legend()
plt.show()

