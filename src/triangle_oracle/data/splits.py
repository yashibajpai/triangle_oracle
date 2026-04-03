import pandas as pd
from sklearn.model_selection import train_test_split


def split_edge_dataset(
    df: pd.DataFrame,
    seed: int = 42,
    train_size: float = 0.70,
    valid_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the edge dataset into train / valid / test.

    The ratios should sum to 1.0.
    """
    total = train_size + valid_size + test_size
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train_size + valid_size + test_size must sum to 1.0")

    train_df, temp_df = train_test_split(df, test_size=(1.0 - train_size), random_state=seed)
    relative_test = test_size / (valid_size + test_size)
    valid_df, test_df = train_test_split(temp_df, test_size=relative_test, random_state=seed)

    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)