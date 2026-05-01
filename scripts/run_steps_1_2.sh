#!/usr/bin/env bash
set -euo pipefail

python src/00_stream_sample.py \
  --only fineweb fineweb_edu \
  --config configs/datasets.yaml \
  --out_dir data/raw \
  --seed 42

python src/01_extract_features.py \
  --raw_dir data/raw \
  --out data/features/features.parquet
