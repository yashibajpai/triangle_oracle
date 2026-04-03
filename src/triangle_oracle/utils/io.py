from pathlib import Path
import json
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not already exist.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: str | Path) -> None:
    """
    Save a Python dictionary as a JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> dict:
    """
    Load a JSON file into a Python dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_npz(path: str | Path, **arrays) -> None:
    """
    Save arrays into a .npz file.

    This mirrors the pattern in the uploaded codebase where
    predictions are saved as .npz and later reloaded by the evaluator.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)