#!/bin/bash
# Evaluate a retrieval checkpoint on the test split.
# Usage: bash scripts/eval.sh <checkpoint_path>

CHECKPOINT=${1:?Usage: bash scripts/eval.sh <checkpoint_path>}

CUDA_VISIBLE_DEVICES=0 python -m opentouch_train.eval \
    --checkpoint "$CHECKPOINT" \
    --data preprocessed_data/train_dataset \
    --split test \
    --batch-size 128 \
    --precision amp
