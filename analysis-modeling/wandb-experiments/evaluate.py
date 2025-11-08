import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import gpytorch

def evaluate_model(model, likelihood, X_test, y_test, scaler_y, epsilon=1e-3):
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(X_test))
        mean = preds.mean.cpu().numpy()
        lower, upper = preds.confidence_region()

    mean_back = scaler_y.inverse_transform(mean.reshape(-1,1)).squeeze()
    y_true_back = scaler_y.inverse_transform(y_test.cpu().numpy().reshape(-1,1)).squeeze()

    mean_final = np.expm1(mean_back) - epsilon
    y_true_final = np.expm1(y_true_back) - epsilon

    rmse = np.sqrt(mean_squared_error(y_true_final, mean_final))
    r2 = r2_score(y_true_final, mean_final)
    return rmse, r2
