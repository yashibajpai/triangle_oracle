from pathlib import Path
import pandas as pd
import networkx as nx


def compute_edge_heaviness(g: nx.Graph) -> pd.DataFrame:
    """
    Build a table where each row is an edge and the target is edge heaviness.

    Edge heaviness = number of common neighbors between endpoints,
    which is exactly the number of triangles containing that edge.
    """
    rows = []

    # Precompute adjacency sets for faster intersection.
    neighbors = {node: set(g.neighbors(node)) for node in g.nodes()}

    for u, v in g.edges():
        common = neighbors[u].intersection(neighbors[v])
        heaviness = len(common)

        rows.append({
            "u": u,
            "v": v,
            "edge_heaviness": heaviness,
        })

    return pd.DataFrame(rows)


def save_edge_dataset(df: pd.DataFrame, path: str | Path) -> None:
    """
    Save the edge-level dataset to CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)