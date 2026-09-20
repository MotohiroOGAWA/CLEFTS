#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.
export OMP_NUM_THREADS=2
python evaluation/branching/prepare_smoke.py
python -m clefts.cli train fragment-tree \
  --train-dir data/train_preprocessing/test/branching_v6_verification/prepared/train_structures \
  --val-dir data/train_preprocessing/test/branching_v6_verification/prepared/validation_structures \
  --output-dir data/train_preprocessing/test/branching_v6_verification/training \
  --mol-encoder-checkpoint data/training/mol_projects/main/mol_encoder_pretrained.pt \
  --epochs 12 --batch-size 4 --max-samples 16 --lr 0.001 --warmup-steps 0 \
  --action-hidden-dim 32 --action-condition-dim 32 --action-threshold 0 \
  --action-top-k 64 --action-max-k 128 --beam-size 32 --max-decode-steps 3 \
  --post-hidden-dim 32 --post-num-layers 1 \
  --validation-interval-steps 4 --validation-fraction 0.5
python evaluation/branching/evaluate_smoke.py
