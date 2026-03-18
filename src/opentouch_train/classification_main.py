"""Main training entry point for OpenTouch classification."""

import glob
import inspect
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from torch import optim

try:
    import wandb
except ImportError:
    wandb = None

from opentouch import create_classification_model
from opentouch.factory import natural_key
from opentouch_train.classification_data import get_classification_data
from opentouch_train.classification_train import train_one_epoch_classification, evaluate_classification
from opentouch_train.classification_params import parse_classification_args
from opentouch_train.distributed import is_master, init_distributed_device, broadcast_object
from opentouch_train.logger import setup_logging
from opentouch_train.scheduler import cosine_lr, const_lr


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def get_latest_checkpoint(path: str):
    checkpoints = glob.glob(os.path.join(path, "**", "*.pt"), recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def _adapt_state_dict_keys(state_dict, distributed):
    """Adapt state dict key prefixes for DDP / non-DDP compatibility."""
    has_module_prefix = next(iter(state_dict)).startswith('module.')
    if distributed and not has_module_prefix:
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    elif not distributed and has_module_prefix:
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


def _build_experiment_name(args) -> str:
    model_name_safe = args.model.replace("/", "-")
    date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    if args.distributed:
        date_str = broadcast_object(args, date_str)
    return "-".join([
        date_str,
        "classify",
        f"model_{model_name_safe}",
        f"lr_{args.lr}",
        f"b_{args.batch_size}",
        f"p_{args.precision}",
    ])


def _create_grad_scaler(device):
    if not (hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")):
        return torch.cuda.amp.GradScaler()
    grad_scaler_sig = inspect.signature(torch.amp.GradScaler)
    if "device" in grad_scaler_sig.parameters:
        return torch.amp.GradScaler(device=device)
    return torch.amp.GradScaler()


def _build_optimizer(model, args):
    def is_excluded(name, param):
        return param.ndim < 2 or "bn" in name or "ln" in name or "bias" in name

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [
        param for name, param in named_parameters
        if is_excluded(name, param) and param.requires_grad
    ]
    rest_params = [
        param for name, param in named_parameters
        if not is_excluded(name, param) and param.requires_grad
    ]

    return optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": args.wd},
        ],
        lr=args.lr,
    )


def main(args):
    args = parse_classification_args(args)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_distributed_device(args)

    if args.name is None:
        args.name = _build_experiment_name(args)

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    args.log_level = logging.INFO
    setup_logging(args.log_path, args.log_level)

    if args.debug:
        args.report_to = ''
        args.save_most_recent = False
        args.save_frequency = 0
        logging.getLogger("torch").setLevel(logging.WARNING)

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        os.makedirs(args.checkpoint_path, exist_ok=True)

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        if is_master(args):
            if args.save_most_recent:
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    resume_from = None
            else:
                resume_from = get_latest_checkpoint(checkpoint_path)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    random_seed(args.seed, 0)

    data = get_classification_data(args)
    assert 'train' in data, 'Training data is required for classification.'

    num_classes = data['train'].dataloader.num_classes
    logging.info(f"Discovered num_classes from data: {num_classes}")

    model = create_classification_model(
        args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        precision=args.precision,
        device=device,
        cache_dir=args.cache_dir,
        enabled_modalities=args.enabled_modalities,
    )

    random_seed(args.seed, args.rank)

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], **ddp_args
        )

    optimizer = _build_optimizer(model, args)

    scaler = None
    if args.precision == "amp":
        scaler = _create_grad_scaler(device)

    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint["epoch"]
            state_dict = checkpoint["state_dict"]
            state_dict = _adapt_state_dict_keys(state_dict, args.distributed)
            model.load_state_dict(state_dict)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            state_dict = _adapt_state_dict_keys(checkpoint, args.distributed)
            model.load_state_dict(state_dict)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    label_smoothing = getattr(args, 'label_smoothing', 0.0)
    loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    scheduler = None
    total_steps = data["train"].dataloader.num_batches * args.epochs
    warmup_steps = (
        int(args.warmup * total_steps) if args.warmup < 1 else int(args.warmup)
    )
    logging.info(f"Warmup steps: {warmup_steps} / {total_steps} total steps")
    if args.lr_scheduler == "cosine":
        scheduler = cosine_lr(optimizer, args.lr, warmup_steps, total_steps)
    elif args.lr_scheduler == "const":
        scheduler = const_lr(optimizer, args.lr, warmup_steps, total_steps)
    else:
        logging.error(
            f'Unknown scheduler, {args.lr_scheduler}. '
            f'Available options are: cosine, const.'
        )
        exit(1)

    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if 'val' in data:
            args.val_sz = data["val"].dataloader.num_samples
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        if is_master(args):
            params_file = os.path.join(args.logs, args.name, "params.txt")
            if os.path.exists(params_file):
                wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    original_model = model

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch_classification(
            model, data, loss, epoch, optimizer, scaler, scheduler, args,
        )
        completed_epoch = epoch + 1

        if 'val' in data:
            evaluate_classification(model, data, completed_epoch, args, num_classes=num_classes)
            if args.distributed:
                torch.distributed.barrier()

        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "num_classes": num_classes,
                "task": args.task,
                "model": args.model,
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0
                and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(
                    args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt"
                )
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(
                    args.checkpoint_path, LATEST_CHECKPOINT_NAME
                )
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

        if args.distributed:
            torch.distributed.barrier()

    if args.wandb and is_master(args):
        wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
