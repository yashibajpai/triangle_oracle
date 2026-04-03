import torch
import torch.nn.functional as F


def log_mse_loss(pred_log: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
    """
    Compare model predictions to log1p(target).

    pred_log is the model output.
    target_raw is the original edge heaviness.
    """
    target_log = torch.log1p(target_raw)
    return F.mse_loss(pred_log, target_log)


def weighted_log_mse_loss(pred_log: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
    """
    Weighted MSE on the log target.

    Heavier edges get more weight because those are more important
    for the downstream learning-augmented algorithm.
    """
    target_log = torch.log1p(target_raw)
    weights = 1.0 + target_log
    return (weights * (pred_log - target_log) ** 2).mean()