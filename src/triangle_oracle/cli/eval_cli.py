import argparse
import numpy as np

from triangle_oracle.laa.eval_oracle import evaluate_with_saved_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--prediction_npz", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)

    # Example:
    # --cutoff_grid 0 1 2 3 5 10 20
    parser.add_argument("--cutoff_grid", type=float, nargs="+", required=True)

    args = parser.parse_args()

    evaluate_with_saved_predictions(
        valid_csv=args.valid_csv,
        test_csv=args.test_csv,
        prediction_npz=args.prediction_npz,
        output_json=args.output_json,
        cutoff_grid=args.cutoff_grid,
    )


if __name__ == "__main__":
    main()