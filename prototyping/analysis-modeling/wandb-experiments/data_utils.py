import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

def load_and_prepare_data(path, lookback_days, temp_horizon_hours, spatial_test_frac):
    df = pd.read_parquet(path)
    df['timestamp_hour'] = pd.to_datetime(df['timestamp_hour'])
    df = df.groupby(['lat', 'lon', 'timestamp_hour'], as_index=False)['PM2_5'].mean()

    latest_time = df['timestamp_hour'].max()
    start_time = latest_time - pd.Timedelta(days=lookback_days)
    df_recent = df[df['timestamp_hour'] >= start_time].copy()

    # spatial split
    unique_locs = df_recent[['lat','lon']].drop_duplicates()
    num_test_locs = int(len(unique_locs) * spatial_test_frac)
    test_locs = unique_locs.sample(n=num_test_locs, random_state=41)

    train_end = latest_time - pd.Timedelta(hours=temp_horizon_hours)

    train_df = df_recent[
        (df_recent['timestamp_hour'] <= train_end) &
        (~df_recent.set_index(['lat','lon']).index.isin(test_locs.set_index(['lat','lon']).index))
    ].copy()
    test_df = df_recent[
        (df_recent['timestamp_hour'] > train_end) |
        (df_recent.set_index(['lat','lon']).index.isin(test_locs.set_index(['lat','lon']).index))
    ].copy()

    train_start = train_df['timestamp_hour'].min()
    for d in [train_df, test_df]:
        d['minutes_since_start'] = (d['timestamp_hour'] - train_start).dt.total_seconds() / 60

    return train_df, test_df


def prepare_tensors(train_df, test_df, epsilon=1e-3, device='cuda'):
    X_train = train_df[['lat','lon','minutes_since_start']].values
    y_train = train_df['PM2_5'].values
    X_test = test_df[['lat','lon','minutes_since_start']].values
    y_test = test_df['PM2_5'].values

    # remove outliers
    mask_train = np.abs(y_train - y_train.mean()) < 3 * y_train.std()
    mask_test = np.abs(y_test - y_test.mean()) < 3 * y_test.std()
    X_train, y_train = X_train[mask_train], y_train[mask_train]
    X_test, y_test = X_test[mask_test], y_test[mask_test]

    # log-transform
    y_train_log = np.log1p(y_train + epsilon)
    y_test_log = np.log1p(y_test + epsilon)

    # scaling
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train_log.reshape(-1, 1))

    X_train_scaled = torch.tensor(scaler_X.transform(X_train), dtype=torch.float32).to(device)
    y_train_scaled = torch.tensor(scaler_y.transform(y_train_log.reshape(-1, 1)), dtype=torch.float32).squeeze().to(device)
    X_test_scaled = torch.tensor(scaler_X.transform(X_test), dtype=torch.float32).to(device)
    y_test_scaled = torch.tensor(scaler_y.transform(y_test_log.reshape(-1, 1)), dtype=torch.float32).squeeze().to(device)

    return (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
            scaler_X, scaler_y)