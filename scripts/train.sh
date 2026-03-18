#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.." || exit

# Single-GPU training
CUDA_VISIBLE_DEVICES=0 python -m opentouch_train.main \
    --train-data preprocessed_data/train_dataset \
    --model OpenTouch-DINOv3-B16-Retrieval \
    --task-type v2t \
    --batch-size 128 \
    --lr 1e-4 \
    --epochs 300 \
    --precision amp \
    --workers 8 \
    --sequence-length 20 \
    --report-to wandb \
    --save-frequency 30 \
    --val-frequency 10 \
    --name single_gpu_v2t
