#!/usr/bin/env bash

python -m src.triangle_oracle.cli.train_cli \
  --train_csv data/processed/train_edges.csv \
  --valid_csv data/processed/valid_edges.csv \
  --test_csv data/processed/test_edges.csv \
  --model_dir outputs/models/mlp_run_01 \
  --prediction_dir outputs/predictions/mlp_run_01 \
  --seed 42 \
  --batch_size 256 \
  --hidden_dims 128 64 \
  --dropout 0.1 \
  --lr 0.001 \
  --epochs 30 \
  --device cpu