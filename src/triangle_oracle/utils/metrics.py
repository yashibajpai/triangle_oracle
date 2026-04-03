import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute standard regression metrics.

    We evaluate on the original scale and also on the log-transformed scale
    elsewhere if needed.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    spear = _safe_spearman(y_true, y_pred)

    return {
        "rmse": rmse,
        "mae": mae,
        "spearman": float(spear),
    }


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """
    Lightweight Spearman correlation using rank conversion.
    """
    if len(a) == 0:
        return 0.0

    a_rank = np.argsort(np.argsort(a)).astype(float)
    b_rank = np.argsort(np.argsort(b)).astype(float)

    a_rank -= a_rank.mean()
    b_rank -= b_rank.mean()

    denom = np.sqrt((a_rank ** 2).sum() * (b_rank ** 2).sum())
    if denom == 0:
        return 0.0

    return float((a_rank * b_rank).sum() / denom)