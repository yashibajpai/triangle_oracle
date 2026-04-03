import numpy as np
import pandas as pd
import networkx as nx


def select_heavy_edges_by_score(edge_df: pd.DataFrame, pred_scores: np.ndarray, score_cutoff: float) -> pd.DataFrame:
    """
    Select edges whose predicted score exceeds the cutoff.
    """
    df = edge_df.copy()
    df["pred_score"] = pred_scores
    return df[df["pred_score"] > score_cutoff].reset_index(drop=True)


def evaluate_heavy_edge_recall(edge_df: pd.DataFrame, pred_scores: np.ndarray, target_col: str = "edge_heaviness", top_frac: float = 0.10) -> dict:
    """
    Evaluate whether the oracle is correctly identifying the heaviest edges.

    This is a useful proxy metric because the downstream algorithm mainly
    benefits if the heavy edges are ranked well.
    """
    df = edge_df.copy()
    df["pred_score"] = pred_scores

    n = len(df)
    if n == 0:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0}

    k = max(1, int(top_frac * n))

    actual_top_idx = df[target_col].to_numpy().argsort()[::-1][:k]
    pred_top_idx = df["pred_score"].to_numpy().argsort()[::-1][:k]

    actual_top_set = set(actual_top_idx.tolist())
    pred_top_set = set(pred_top_idx.tolist())

    overlap = len(actual_top_set.intersection(pred_top_set))

    precision = overlap / k
    recall = overlap / k

    return {
        "precision_at_k": float(precision),
        "recall_at_k": float(recall),
    }


def learned_oracle_summary(edge_df: pd.DataFrame, pred_scores: np.ndarray, score_cutoff: float) -> dict:
    """
    Summarize how many edges would be treated as oracle-selected.
    """
    chosen = select_heavy_edges_by_score(edge_df, pred_scores, score_cutoff)
    total = len(edge_df)

    return {
        "num_selected_edges": int(len(chosen)),
        "fraction_selected_edges": float(len(chosen) / total) if total > 0 else 0.0,
    }