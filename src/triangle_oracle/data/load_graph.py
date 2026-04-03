from pathlib import Path
import pandas as pd
import networkx as nx


def load_edge_list(path: str | Path, source_col: str = "u", target_col: str = "v") -> nx.Graph:
    """
    Load a graph from a CSV file with edge columns.

    Expected format:
        u,v
        P53,MDM2
        MDM2,RB1
        ...

    We build an undirected simple graph because triangle counting
    usually assumes an undirected graph here.
    """
    df = pd.read_csv(path)

    if source_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Expected columns '{source_col}' and '{target_col}' in {path}")

    g = nx.Graph()
    for _, row in df.iterrows():
        u = row[source_col]
        v = row[target_col]

        # Avoid self-loops for triangle counting unless your project explicitly needs them.
        if u == v:
            continue

        g.add_edge(u, v)

    return g