"""Training parameters for OpenTouch cross-modal retrieval."""

import argparse


def get_default_params(model_name):
    return {"lr": 1.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to dataset directory. Comma-separated for multi-dataset co-training.",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation data (if separate from train-data). By default, val split from train-data is used.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=20,
        help="Temporal sequence length per sample.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Sliding window stride. Defaults to --sequence-length (non-overlapping).",
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
        default="OpenTouch-DINOv3-B16-Retrieval",
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
        "--task-type",
        type=str,
        default="v2t",
        help=(
            "Cross-modal retrieval task. "
            "Valid: v2t, p2t, v2p, vp2t, tp2v, vt2p."
        ),
    )

    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="Log files on local master, otherwise global master only.",
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
        "--batch-size", type=int, default=32, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--epochs-cooldown", type=int, default=None,
        help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=float, default=0.05,
        help="Warmup: fraction of total steps if < 1 (e.g. 0.05 = 5%%), or absolute step count if >= 1.",
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.",
    )
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default='cosine',
        help="LR scheduler. One of: 'cosine', 'const', 'const-cooldown'.",
    )
    parser.add_argument(
        "--lr-cooldown-end", type=float, default=0.0,
        help="End learning rate for cooldown schedule.",
    )
    parser.add_argument(
        "--lr-cooldown-power", type=float, default=1.0,
        help="Power for polynomial cooldown schedule.",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=20, help="How often to save epoch checkpoints (0 = disabled, only latest is saved)."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=True,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--val-frequency", type=int, default=20, help="How often to run evaluation with val data (in epochs)."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="Path to latest checkpoint (default: none).",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp",
        help="Floating point precision.",
    )

    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="Calculate loss w/ local features @ global.",
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="Enable full distributed gradient for feature gather.",
    )

    parser.add_argument(
        "--accum-freq", type=int, default=1, help="Update the model every --accum-freq steps."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip.",
    )
    parser.add_argument(
        "--torchcompile",
        default=False,
        action='store_true',
        help="torch.compile() the model.",
    )
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
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
        "--report-to",
        default='wandb',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard'].",
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb.",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default='opentouch',
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged.",
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, copy the entire codebase on the log directory.",
    )

    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one.",
    )

    args = parser.parse_args(args)

    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
