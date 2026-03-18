"""CLI argument parser for OpenTouch classification training."""

import argparse


def parse_classification_args(args):
    parser = argparse.ArgumentParser(description="OpenTouch Classification Training")

    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=20,
        help="Temporal sequence length per sample.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Target image size (H W).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for validation split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for test split.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="OpenTouch-DINOv3-B16-Classify",
        help="Name of the model config.",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Path to pretrained model checkpoint.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override system default cache path for model file downloads.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="action",
        choices=["action", "grip"],
        help="Classification task to train on.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=["visual", "tactile"],
        choices=["visual", "tactile", "pose"],
        help="Input modalities to use.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor for cross-entropy loss.",
    )

    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=500, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=float, default=0.05,
        help="Warmup: fraction of total steps if < 1, or absolute step count if >= 1.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "const", "step"],
        help="LR scheduler type.",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=20,
        help="How often to save epoch checkpoints (0 = disabled, only latest is saved).",
    )
    parser.add_argument(
        "--val-frequency", type=int, default=20,
        help="How often to run evaluation with val data (in epochs).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="Path to latest checkpoint (default: none).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="amp",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=1.0, help="Gradient clip norm.",
    )

    parser.add_argument(
        "--device", default="cuda", type=str, help="Accelerator to use.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Default random seed.",
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="URL used to set up distributed training.",
    )
    parser.add_argument(
        "--dist-backend",
        default=None,
        type=str,
        help='Distributed backend. "nccl" for GPU, "hccl" for Ascend NPU.',
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank.",
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.",
    )

    parser.add_argument(
        "--report-to",
        default='wandb',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard'].",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default='opentouch-classification',
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=50,
        help="Log every n steps to console/wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="Log files on local master, otherwise global master only.",
    )

    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=True,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one.",
    )

    parsed = parser.parse_args(args)
    parsed.enabled_modalities = parsed.modalities
    return parsed
