from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from triangle_oracle.training.dataset import EdgeHeavinessDataset
from triangle_oracle.training.engine import train_one_epoch, evaluate
from triangle_oracle.models.mlp_oracle import MLPOracle
from triangle_oracle.models.losses import weighted_log_mse_loss
from triangle_oracle.utils.io import ensure_dir, save_json, save_npz
from triangle_oracle.utils.metrics import regression_metrics
from triangle_oracle.utils.seed import set_seed


def run_training(
    train_csv: str,
    valid_csv: str,
    test_csv: str,
    model_dir: str,
    prediction_dir: str,
    seed: int = 42,
    batch_size: int = 256,
    hidden_dims: list[int] = [128, 64],
    dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 30,
    device: str = "cpu",
):
    """
    End-to-end training for the MLP oracle.

    This function:
    1. loads train/valid/test data,
    2. trains the model,
    3. saves the best checkpoint,
    4. saves valid/test predictions to .npz.
    """
    set_seed(seed)

    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    # Load tabular datasets.
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    test_df = pd.read_csv(test_csv)

    feature_cols = [c for c in train_df.columns if c not in {"u", "v", "edge_heaviness"}]

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["edge_heaviness"].to_numpy(dtype=np.float32)

    X_valid = valid_df[feature_cols].to_numpy(dtype=np.float32)
    y_valid = valid_df["edge_heaviness"].to_numpy(dtype=np.float32)

    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["edge_heaviness"].to_numpy(dtype=np.float32)

    train_ds = EdgeHeavinessDataset(X_train, y_train)
    valid_ds = EdgeHeavinessDataset(X_valid, y_valid)
    test_ds = EdgeHeavinessDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = MLPOracle(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = weighted_log_mse_loss

    best_valid_loss = float("inf")
    best_checkpoint_path = None

    ensure_dir(model_dir)
    ensure_dir(prediction_dir)

    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        valid_loss, valid_pred, valid_true = evaluate(model, valid_loader, device, loss_fn)
        test_loss, test_pred, test_true = evaluate(model, test_loader, device, loss_fn)

        valid_metrics = regression_metrics(valid_true, valid_pred)
        test_metrics = regression_metrics(test_true, test_pred)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "test_loss": test_loss,
            "valid_rmse": valid_metrics["rmse"],
            "valid_mae": valid_metrics["mae"],
            "valid_spearman": valid_metrics["spearman"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_spearman": test_metrics["spearman"],
        }
        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"valid_loss={valid_loss:.4f} "
            f"test_loss={test_loss:.4f} "
            f"valid_spearman={valid_metrics['spearman']:.4f}"
        )

        # Save the best checkpoint based on validation loss.
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_checkpoint_path = Path(model_dir) / "best_model.pt"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_cols": feature_cols,
                    "hidden_dims": hidden_dims,
                    "dropout": dropout,
                    "seed": seed,
                },
                best_checkpoint_path,
            )

            # Save predictions in the same spirit as the old codebase:
            # valid_output and test_output inside a .npz file.
            save_npz(
                Path(prediction_dir) / "best_predictions.npz",
                valid_output=valid_pred.reshape(-1, 1),
                test_output=test_pred.reshape(-1, 1),
                valid_targets=valid_true.reshape(-1, 1),
                test_targets=test_true.reshape(-1, 1),
            )

    save_json(
        {
            "seed": seed,
            "batch_size": batch_size,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "lr": lr,
            "epochs": epochs,
            "feature_cols": feature_cols,
            "best_valid_loss": best_valid_loss,
            "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else None,
            "history": history,
        },
        Path(model_dir) / "train_summary.json",
    )