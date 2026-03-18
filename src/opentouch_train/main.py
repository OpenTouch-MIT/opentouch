"""Main training entry point for OpenTouch cross-modal retrieval."""

import copy
import glob
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

from opentouch import create_model_and_transforms, create_loss
from opentouch.factory import natural_key
from opentouch_train.data import get_data
from opentouch_train.distributed import is_master, init_distributed_device, broadcast_object
from opentouch_train.logger import setup_logging
from opentouch_train.params import parse_args
from opentouch_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
from opentouch_train.train import train_one_epoch, evaluate


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def get_latest_checkpoint(path: str):
    checkpoints = glob.glob(path + '**/*.pt', recursive=True)
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


def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_distributed_device(args)

    if args.name is None:
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

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

    if args.copy_codebase:
        copy_codebase(args)

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    random_seed(args.seed, 0)

    from opentouch_train.data import parse_task
    query_mods, target_mods = parse_task(args.task_type)
    enabled_modalities = list(set(query_mods) | set(target_mods))

    model_kwargs = {
        'enabled_modalities': enabled_modalities,
    }

    model, _, _ = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        cache_dir=args.cache_dir,
        **model_kwargs,
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

    optimizer = None
    scaler = None

    if args.train_data:
        exclude = lambda n, p: (
            p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        )
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad
        ]
        rest_params = [
            p for n, p in named_parameters if include(n, p) and p.requires_grad
        ]

        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )

        if is_master(args):
            defaults = copy.deepcopy(optimizer.defaults)
            defaults['weight_decay'] = args.wd
            defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
            logging.info(
                f'Created AdamW optimizer: {defaults}'
            )

        scaler = None
        if args.precision == "amp":
            try:
                scaler = torch.amp.GradScaler(device=device)
            except (AttributeError, TypeError):
                scaler = torch.cuda.amp.GradScaler()

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

    data = get_data(args, epoch=start_epoch)
    assert len(data), 'At least one train or eval dataset must be specified.'

    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (
            (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        )
        warmup_steps = (
            int(args.warmup * total_steps) if args.warmup < 1 else int(args.warmup)
        )
        logging.info(f"Warmup steps: {warmup_steps} / {total_steps} total steps")
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, warmup_steps, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, warmup_steps, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None, \
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (
                (data["train"].dataloader.num_batches // args.accum_freq)
                * args.epochs_cooldown
            )
            scheduler = const_lr_cooldown(
                optimizer, args.lr, warmup_steps, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end,
            )
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. '
                f'Available options are: cosine, const, const-cooldown.'
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
    if args.torchcompile:
        logging.info('Compiling model...')
        if args.grad_checkpointing and args.distributed:
            logging.info(
                'Disabling DDP dynamo optimizer when grad checkpointing enabled.'
            )
            torch._dynamo.config.optimize_ddp = False

        filter_prefixes = (
            "torch._dynamo",
            "torch._inductor",
            "torch._functorch",
            "torch._utils_internal",
            "torch.fx",
        )
        for name in logging.root.manager.loggerDict:
            if name.startswith(filter_prefixes):
                logging.getLogger(name).setLevel(logging.WARNING)

        model = torch.compile(original_model)

    if 'train' not in data:
        evaluate(model, data, start_epoch, args, tb_writer=None)
        return

    loss = create_loss(args)

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(
            model, data, loss, epoch, optimizer, scaler, scheduler, None, args,
            tb_writer=None,
        )
        completed_epoch = epoch + 1

        if 'val' in data:
            evaluate(model, data, completed_epoch, args, tb_writer=None)
            if args.distributed:
                torch.distributed.barrier()

        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "task_type": args.task_type,
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


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. "
            f"Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(
        current_code_path,
        new_code_path,
        ignore=ignore_patterns('log', 'logs', 'wandb'),
    )
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
