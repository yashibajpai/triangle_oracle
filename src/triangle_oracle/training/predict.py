from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from triangle_oracle.training.dataset import EdgeHeavinessDataset
from triangle_oracle.models.mlp_oracle import MLPOracle
from triangle_oracle.utils.io import save_npz


@torch.no_grad()
def run_prediction(
    checkpoint_path: str,
    valid_csv: str,
    test_csv: str,
    output_path: str,
    batch_size: int = 256,
    device: str = "cpu",
):
    """
    Load a trained model and write valid/test predictions to a .npz file.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    feature_cols = ckpt["feature_cols"]
    hidden_dims = ckpt["hidden_dims"]
    dropout = ckpt["dropout"]

    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    valid_df = pd.read_csv(valid_csv)
    test_df = pd.read_csv(test_csv)

    X_valid = valid_df[feature_cols].to_numpy(dtype=np.float32)
    y_valid = valid_df["edge_heaviness"].to_numpy(dtype=np.float32)

    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["edge_heaviness"].to_numpy(dtype=np.float32)

    valid_ds = EdgeHeavinessDataset(X_valid, y_valid)
    test_ds = EdgeHeavinessDataset(X_test, y_test)

    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = MLPOracle(
        input_dim=len(feature_cols),
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    valid_pred = []
    for batch in valid_loader:
        x = batch["x"].to(device)
        pred_log = model(x)
        pred_raw = torch.expm1(pred_log).cpu().numpy()
        valid_pred.append(pred_raw)

    test_pred = []
    for batch in test_loader:
        x = batch["x"].to(device)
        pred_log = model(x)
        pred_raw = torch.expm1(pred_log).cpu().numpy()
        test_pred.append(pred_raw)

    valid_pred = np.concatenate(valid_pred)
    test_pred = np.concatenate(test_pred)

    save_npz(
        output_path,
        valid_output=valid_pred.reshape(-1, 1),
        test_output=test_pred.reshape(-1, 1),
        valid_targets=y_valid.reshape(-1, 1),
        test_targets=y_test.reshape(-1, 1),
    )