#!/usr/bin/env bash

python -m src.triangle_oracle.cli.predict_cli \
  --checkpoint_path outputs/models/mlp_run_01/best_model.pt \
  --valid_csv data/processed/valid_edges.csv \
  --test_csv data/processed/test_edges.csv \
  --output_path outputs/predictions/mlp_run_01/predictions.npz \
  --batch_size 256 \
  --device cpu