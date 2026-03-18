#!/bin/bash
# Debug classification training (2 epochs, no wandb)

CUDA_VISIBLE_DEVICES=0 python -m opentouch_train.classification_main \
    --train-data preprocessed_data/classification_test \
    --model OpenTouch-DINOv3-B16-Classify \
    --task action \
    --modalities visual tactile \
    --batch-size 32 \
    --lr 3e-3 \
    --epochs 2 \
    --sequence-length 20 \
    --workers 2 \
    --precision amp \
    --debug
