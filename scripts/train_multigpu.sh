#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.." || exit

# Multi-GPU training via torchrun
# Adjust --nproc_per_node and CUDA_VISIBLE_DEVICES to match your setup
CUDA_VISIBLE_DEVICES=0,3,7 torchrun --nproc_per_node 3 \
    -m opentouch_train.main \
    --train-data preprocessed_data/train_dataset \
    --model OpenTouch-DINOv3-B16-Retrieval \
    --task-type v2t \
    --batch-size 128 \
    --lr 1e-3 \
    --epochs 300 \
    --precision amp \
    --workers 4 \
    --sequence-length 20 \
    --report-to wandb \
    --save-frequency 30 \
    --val-frequency 15 \
    --local-loss \
    --gather-with-grad \
    --name multi_gpu_v2t
