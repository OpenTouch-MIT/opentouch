#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.." || exit

# Debug training: 2 epochs, no wandb, no checkpoints
CUDA_VISIBLE_DEVICES=0 python -m opentouch_train.main \
    --train-data preprocessed_data/train_dataset \
    --model OpenTouch-DINOv3-B16-Retrieval \
    --task-type v2t \
    --batch-size 32 \
    --lr 1e-4 \
    --epochs 2 \
    --workers 2 \
    --sequence-length 20 \
    --debug
