#!/usr/bin/env bash

SONOSCENE360_ROOT=/path/to/SonoScene360

if [[ -z $SONOSCENE360_ROOT ]]; then
  echo "Usage: $0 <sonoscene360-root>" >&2
  exit 2
fi

python inference.py \
  --dataset-root "$SONOSCENE360_ROOT" \
  --scenes-root outputs/sonoscene360 \
  --output-root outputs/sonoscene360_inference \
  --all-scenes

python evaluate.py \
  --dataset-root "$SONOSCENE360_ROOT" \
  --predictions-root outputs/sonoscene360_inference \
  --output-root outputs/sonoscene360_evaluation \
  --all-scenes
