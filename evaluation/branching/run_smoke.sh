#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.
export OMP_NUM_THREADS=2
python evaluation/branching/prepare_smoke.py
python -m clefts.cli train fragment-tree \
  --train-dir data/train_preprocessing/test/fragment_tree_redesign_verification/prepared/train_structures \
  --val-dir data/train_preprocessing/test/fragment_tree_redesign_verification/prepared/validation_structures \
  --output-dir data/train_preprocessing/test/fragment_tree_redesign_verification/training \
  --mol-encoder-checkpoint data/training/mol_projects/main/mol_encoder_pretrained.pt \
  --epochs 12 --batch-size 4 --max-samples 16 --lr 0.001 --warmup-steps 0 \
  --action-hidden-dim 32 --action-main-adduct-dim 32 \
  --branch-path-threshold 0 --max-fragment-nodes 100 \
  --post-hidden-dim 32 --post-num-layers 1 \
  --validation-interval-steps 4 --validation-fraction 0.5
python evaluation/branching/evaluate_smoke.py
