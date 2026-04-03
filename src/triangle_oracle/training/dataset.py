import numpy as np
import torch
from torch.utils.data import Dataset


class EdgeHeavinessDataset(Dataset):
    """
    A PyTorch Dataset for edge-level prediction.

    Inputs:
        X: feature matrix, shape [num_edges, num_features]
        y: target array, shape [num_edges]
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict:
        return {
            "x": self.X[idx],
            "y": self.y[idx],
        }