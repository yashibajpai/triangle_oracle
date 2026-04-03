import time
import networkx as nx


def exact_triangle_count(g: nx.Graph) -> int:
    """
    Exact global triangle count for benchmarking.

    networkx.triangles returns per-node triangle participation counts.
    Summing and dividing by 3 gives the total number of triangles.
    """
    tri_per_node = nx.triangles(g)
    return sum(tri_per_node.values()) // 3


def run_baseline(g: nx.Graph) -> dict:
    """
    Benchmark the exact baseline.
    """
    start = time.time()
    total_triangles = exact_triangle_count(g)
    elapsed = time.time() - start

    return {
        "triangle_count": int(total_triangles),
        "runtime_sec": float(elapsed),
    }