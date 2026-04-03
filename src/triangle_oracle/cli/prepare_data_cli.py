import argparse
from pathlib import Path

from triangle_oracle.data.load_graph import load_edge_list
from triangle_oracle.data.make_edge_dataset import compute_edge_heaviness
from triangle_oracle.data.features import build_edge_features, merge_features_and_targets
from triangle_oracle.data.splits import split_edge_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Raw edge list CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Where processed files will be saved")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading graph...")
    g = load_edge_list(args.input_csv)

    print("Computing edge heaviness labels...")
    edge_df = compute_edge_heaviness(g)

    print("Building edge features...")
    feature_df = build_edge_features(g, edge_df)

    print("Merging features and labels...")
    full_df = merge_features_and_targets(feature_df, edge_df)

    print("Splitting dataset...")
    train_df, valid_df, test_df = split_edge_dataset(full_df)

    train_df.to_csv(output_dir / "train_edges.csv", index=False)
    valid_df.to_csv(output_dir / "valid_edges.csv", index=False)
    test_df.to_csv(output_dir / "test_edges.csv", index=False)

    print(f"Saved processed data to {output_dir}")


if __name__ == "__main__":
    main()