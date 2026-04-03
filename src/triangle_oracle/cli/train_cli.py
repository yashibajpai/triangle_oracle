import argparse

from triangle_oracle.training.train import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--valid_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)

    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--prediction_dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    run_training(
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        test_csv=args.test_csv,
        model_dir=args.model_dir,
        prediction_dir=args.prediction_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
    )


if __name__ == "__main__":
    main()