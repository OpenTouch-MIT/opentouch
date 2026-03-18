#!/bin/bash
# Evaluate a classification checkpoint on the test split.
# All parameters (model, task, modalities, data path, etc.) are auto-detected from params.txt.
# Usage: bash scripts/eval_classifier.sh <checkpoint_path>

CHECKPOINT=${1:?Usage: bash scripts/eval_classifier.sh <checkpoint_path>}

CUDA_VISIBLE_DEVICES=0 python -m opentouch_train.classification_eval \
    --checkpoint "$CHECKPOINT" \
    --split test
