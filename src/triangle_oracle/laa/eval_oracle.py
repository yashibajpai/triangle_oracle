from pathlib import Path
import json
import numpy as np
import pandas as pd

from triangle_oracle.laa.learned_triangle import (
    evaluate_heavy_edge_recall,
    learned_oracle_summary,
)


def search_best_cutoff(valid_df: pd.DataFrame, valid_scores: np.ndarray, cutoff_grid: list[float]) -> dict:
    """
    Search the best score cutoff on validation.

    For now, we optimize recall/precision overlap on heavy edges.
    Later, you can replace this objective with downstream triangle
    counting speed/accuracy.
    """
    best = None
    best_score = -1.0

    for cutoff in cutoff_grid:
        metrics = evaluate_heavy_edge_recall(valid_df, valid_scores)
        summary = learned_oracle_summary(valid_df, valid_scores, cutoff)

        # Simple objective:
        # prefer good heavy-edge identification while not selecting every edge.
        objective = metrics["precision_at_k"] + metrics["recall_at_k"] - 0.1 * summary["fraction_selected_edges"]

        result = {
            "score_cutoff": float(cutoff),
            "objective": float(objective),
            **metrics,
            **summary,
        }

        if objective > best_score:
            best_score = objective
            best = result

    return best


def evaluate_with_saved_predictions(
    valid_csv: str,
    test_csv: str,
    prediction_npz: str,
    output_json: str,
    cutoff_grid: list[float],
):
    """
    Load saved predictions from a .npz file and run validation/test evaluation.

    This follows the same high-level pattern as the uploaded repo:
    predictions are saved separately, then consumed by a learned evaluator.
    """
    valid_df = pd.read_csv(valid_csv)
    test_df = pd.read_csv(test_csv)

    pred_data = np.load(prediction_npz)
    valid_scores = pred_data["valid_output"].squeeze()
    test_scores = pred_data["test_output"].squeeze()

    best_valid = search_best_cutoff(valid_df, valid_scores, cutoff_grid)
    chosen_cutoff = best_valid["score_cutoff"]

    test_metrics = evaluate_heavy_edge_recall(test_df, test_scores)
    test_summary = learned_oracle_summary(test_df, test_scores, chosen_cutoff)

    output = {
        "validation_selection": best_valid,
        "test_results": {
            "score_cutoff": float(chosen_cutoff),
            **test_metrics,
            **test_summary,
        },
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))