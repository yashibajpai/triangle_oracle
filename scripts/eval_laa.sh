#!/usr/bin/env bash

python -m src.triangle_oracle.cli.eval_cli \
  --valid_csv data/processed/valid_edges.csv \
  --test_csv data/processed/test_edges.csv \
  --prediction_npz outputs/predictions/mlp_run_01/predictions.npz \
  --output_json outputs/eval/mlp_run_01_eval.json \
  --cutoff_grid 0 1 2 3 5 10 20