from tqdm import trange
import torch
import gpytorch
import wandb
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from evaluate import evaluate_model
import numpy as np

def train_model(
    model,
    likelihood,
    optimizer,
    mll,
    X_train,
    y_train,
    X_val,
    y_val,
    config,
    scaler_y,
    device='cuda'
):
    """
    Optimized variational GP training for sweeps.
    Uses mini-batching, GPU-efficient evaluation, reduced logging, and tqdm progress bar.
    """

    model.to(device).train()
    likelihood.to(device).train()

    batch_size = config.get("batch_size", 512)
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    num_iters = config["iterations"]
    log_interval = config.get("log_interval", 100)  # reduced logging
    patience = config.get("patience", 50)

    best_val_rmse = float("inf")
    trigger = 0
    best_state = None

    wandb.define_metric("iteration")
    wandb.define_metric("train_loss", step_metric="iteration")
    wandb.define_metric("val_rmse", step_metric="iteration")
    wandb.define_metric("train_rmse", step_metric="iteration")

    print("Commencing training...")
    # --- tqdm over iterations ---
    for i in trange(num_iters, desc="Training GPR", dynamic_ncols=True, disable=wandb.run is not None):  #
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            output = model(xb)
            loss = -mll(output, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * xb.size(0)

        # --- logging & validation ---
        if (i + 1) % log_interval == 0 or i == num_iters - 1:
            model.eval()
            likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                # batched evaluation
                def batched_eval(X, y, batch_size=1024):
                    preds = []
                    for start in range(0, X.size(0), batch_size):
                        end = start + batch_size
                        pred = likelihood(model(X[start:end]))
                        preds.append(pred.mean.cpu())
                    mean = torch.cat(preds).numpy()
                    y_true = y.cpu().numpy()
                    # inverse transform & exponentiate
                    mean_back = scaler_y.inverse_transform(mean.reshape(-1,1)).squeeze()
                    y_true_back = scaler_y.inverse_transform(y_true.reshape(-1,1)).squeeze()
                    return (np.expm1(mean_back) - 1e-3, np.expm1(y_true_back) - 1e-3)

                train_pred, train_true = batched_eval(X_train, y_train)
                val_pred, val_true = batched_eval(X_val, y_val)

            train_rmse = np.sqrt(np.mean((train_pred - train_true)**2))
            val_rmse = np.sqrt(np.mean((val_pred - val_true)**2))

            train_loss = epoch_loss / X_train.size(0)

            wandb.log({
                "iteration": i + 1,
                "train_loss": train_loss,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "learning_rate": optimizer.param_groups[0]["lr"]
            })

            # --- Early stopping ---
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                trigger = 0
                best_state = (model.state_dict(), likelihood.state_dict())
            else:
                trigger += 1
                if trigger >= patience:
                    print(f"Early stopping at iteration {i+1}")
                    break

            model.train()
            likelihood.train()

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state[0])
        likelihood.load_state_dict(best_state[1])
    else:
        print("Warning: no improvement observed; returning last model state.")

    # Save checkpoint
    ckpt_dir = Path("/storage/bln-aq/analysis-modeling/wandb-experiments/wandb-checkpoint-cache")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{wandb.run.name or 'unnamed-run'}_final.pt"

    torch.save({
        "model_state": model.state_dict(),
        "likelihood_state": likelihood.state_dict(),
        "val_rmse": best_val_rmse,
        "config": config
    }, ckpt_path)
    wandb.log({"checkpoint_path": str(ckpt_path)})
    print(f"Final checkpoint saved to: {ckpt_path}")

    return model, likelihood
