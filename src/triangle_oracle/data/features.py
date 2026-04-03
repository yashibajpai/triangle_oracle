import pandas as pd
import networkx as nx
import numpy as np


def build_edge_features(g: nx.Graph, edge_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create hand-crafted features for each edge.

    Keep features cheap and legal:
    - endpoint degrees
    - degree combinations
    - clustering coefficients
    - simple graph statistics

    Avoid exact common-neighbor count here because that would leak the label.
    """
    degree = dict(g.degree())
    clustering = nx.clustering(g)

    rows = []
    for _, row in edge_df.iterrows():
        u = row["u"]
        v = row["v"]

        du = degree[u]
        dv = degree[v]
        cu = clustering[u]
        cv = clustering[v]

        rows.append({
            "u": u,
            "v": v,

            # Raw endpoint stats
            "deg_u": du,
            "deg_v": dv,

            # Symmetric degree features
            "deg_min": min(du, dv),
            "deg_max": max(du, dv),
            "deg_sum": du + dv,
            "deg_prod": du * dv,
            "deg_absdiff": abs(du - dv),

            # Node clustering information
            "clust_u": cu,
            "clust_v": cv,
            "clust_sum": cu + cv,
            "clust_absdiff": abs(cu - cv),
        })

    return pd.DataFrame(rows)


def merge_features_and_targets(feature_df: pd.DataFrame, edge_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge features with target labels.

    We merge on (u, v), assuming both dataframes list the same edges.
    """
    merged = feature_df.merge(edge_df, on=["u", "v"], how="inner")
    return merged


def extract_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Return:
        X: feature matrix
        feature_cols: names of numeric feature columns
    """
    exclude = {"u", "v", "edge_heaviness"}
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    return X, feature_cols