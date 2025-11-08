# train.py
import torch
import gpytorch
import wandb
from tqdm import trange
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def train_model(model, likelihood, optimizer, mll, X_train, y_train, X_val, y_val, config, scaler_y, device='cpu'):
    model.train()
    likelihood.train()

    # Create dataloaders for minibatch training (VariationalGP supports minibatch ELBO)
    batch_size = config.get('batch_size', None)
    if batch_size is None:
        # full-batch
        train_loader = [(X_train, y_train)]
    else:
        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_val = float('inf')
    trigger = 0
    best_state = None

    wandb.define_metric("iteration")
    wandb.define_metric("train_loss", step_metric="iteration")
    wandb.define_metric("val_loss", step_metric="iteration")

    num_iters = config['iterations']
    log_interval = config.get('log_interval', 25)

    for i in trange(num_iters, desc="Training GPR"):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            xb, yb = batch
            output = model(xb)
            loss = -mll(output, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() if isinstance(loss, torch.Tensor) else float(loss)

        # log/validate
        if (i+1) % log_interval == 0 or i == num_iters - 1:
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                val_out = model(X_val)
                val_loss = -mll(val_out, y_val).item()
                # also compute RMSE on transformed scale
                preds = likelihood(val_out)
                mean = preds.mean.cpu().numpy()
                mean_back = scaler_y.inverse_transform(mean.reshape(-1,1)).squeeze()
                y_true_back = scaler_y.inverse_transform(y_val.cpu().numpy().reshape(-1,1)).squeeze()
                mean_final = np.expm1(mean_back) - config.get('epsilon', 1e-3)
                y_true_final = np.expm1(y_true_back) - config.get('epsilon', 1e-3)
                rmse = np.sqrt(mean_squared_error(y_true_final, mean_final))
            model.train()
            wandb.log({
                "iteration": i+1,
                "train_loss": epoch_loss,
                "val_loss": val_loss,
                "val_rmse": rmse,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

            # early stopping & checkpoint
            if val_loss < best_val:
                best_val = val_loss
                trigger = 0
                best_state = (model.state_dict(), likelihood.state_dict())
                # save to wandb artifact or checkpoint
                torch.save({
                    'model_state': best_state[0],
                    'likelihood_state': best_state[1],
                    'val_loss': best_val,
                    'config': config
                }, f"checkpoint_iter_{i+1}.pt")
                wandb.save(f"checkpoint_iter_{i+1}.pt")
            else:
                trigger += 1
                if trigger >= config.get('patience', 50):
                    print(f"Early stopping at iteration {i+1}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state[0])
        likelihood.load_state_dict(best_state[1])
    else:
        print("Warning: no improvement observed; returning last model state.")

    return model, likelihood
