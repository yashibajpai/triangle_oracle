import torch
import torch.nn as nn


class MLPOracle(nn.Module):
    """
    Simple MLP for edge heaviness prediction.

    The model predicts a single scalar per edge.
    We train it on log1p(edge_heaviness) because the target is highly skewed.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.1):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers.
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final regression output.
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return shape: [batch_size]
        """
        return self.net(x).squeeze(-1)