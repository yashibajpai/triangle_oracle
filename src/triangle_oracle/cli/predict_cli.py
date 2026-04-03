import argparse

from triangle_oracle.training.predict import run_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--valid_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_prediction(
        checkpoint_path=args.checkpoint_path,
        valid_csv=args.valid_csv,
        test_csv=args.test_csv,
        output_path=args.output_path,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()