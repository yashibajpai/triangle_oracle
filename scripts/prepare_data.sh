#!/usr/bin/env bash

python -m src.triangle_oracle.cli.prepare_data_cli \
  --input_csv data/raw/ppi_edges.csv \
  --output_dir data/processed