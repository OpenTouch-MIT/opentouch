#!/bin/bash
# Train action classifier with visual+tactile modalities

CUDA_VISIBLE_DEVICES=6 python -m opentouch_train.classification_main \
    --train-data preprocessed_data/classification_test \
    --model OpenTouch-DINOv3-B16-Classify \
    --task action \
    --modalities visual tactile \
    --batch-size 64 \
    --lr 3e-3 \
    --epochs 500 \
    --sequence-length 20 \
    --workers 8 \
    --precision amp \
    --lr-scheduler cosine \
    --grad-clip-norm 1.0 \
    --val-frequency 10 \
    --wandb-project-name opentouch-classification \
    --report-to wandb
