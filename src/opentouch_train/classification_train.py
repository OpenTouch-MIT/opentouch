"""Training loop for OpenTouch classification"""

import json
import logging
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from opentouch import get_input_dtype
from opentouch.classification_metrics import compute_classification_metrics
from opentouch_train.data import MODALITY_TO_BATCH_KEY
from opentouch_train.distributed import is_master
from opentouch_train.precision import get_autocast
from opentouch_train.train import AverageMeter, unwrap_model, backward


def _extract_batch_tensors(batch, enabled_modalities, device, input_dtype):
    """Move modality tensors from batch dict to device."""
    tensors = {}
    for mod in enabled_modalities:
        batch_key = MODALITY_TO_BATCH_KEY[mod]
        if batch_key in batch:
            t = batch[batch_key]
            if input_dtype is not None:
                t = t.to(device=device, dtype=input_dtype, non_blocking=True)
            else:
                t = t.to(device=device, non_blocking=True)
            tensors[batch_key] = t
    return tensors


def _extract_labels(batch, device):
    """Extract label tensor from batch to device."""
    return batch["label"].to(device=device, non_blocking=True)


def _is_log_step(step_idx: int, num_batches: int, log_every_n_steps: int) -> bool:
    return (step_idx % log_every_n_steps == 0) or ((step_idx + 1) == num_batches)


def _is_val_epoch(epoch: int, val_frequency: int, total_epochs: int) -> bool:
    if not val_frequency:
        return True
    return (epoch % val_frequency) == 0 or epoch == total_epochs


def train_one_epoch_classification(model, data, loss_fn, epoch, optimizer, scaler, scheduler, args):
    """Run one training epoch for classification."""
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()

    enabled_modalities = args.enabled_modalities

    data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader
    num_batches = dataloader.num_batches

    losses_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    pbar = tqdm(
        enumerate(dataloader),
        total=num_batches,
        desc=f"Epoch {epoch}",
        disable=not is_master(args),
    )
    for i, batch in pbar:
        step = num_batches * epoch + i

        if not args.skip_scheduler and scheduler is not None:
            scheduler(step)

        batch_tensors = _extract_batch_tensors(batch, enabled_modalities, device, input_dtype)
        labels = _extract_labels(batch, device)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            logits = model(**batch_tensors)
            total_loss = loss_fn(logits, labels)

        backward(total_loss, scaler)

        if scaler is not None:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        batch_time_m.update(time.time() - end)
        end = time.time()

        batch_size = len(labels)
        losses_m.update(total_loss.item(), batch_size)

        if is_master(args) and _is_log_step(i, num_batches, args.log_every_n_steps):
            samples_per_second = args.batch_size * args.world_size / batch_time_m.val

            pbar.set_postfix(
                loss=f"{losses_m.avg:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                sps=f"{samples_per_second:.0f}",
            )

            log_data = {
                "train/loss": losses_m.val,
                "train/loss_avg": losses_m.avg,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/data_time": data_time_m.val,
                "train/batch_time": batch_time_m.val,
                "train/samples_per_second": samples_per_second,
            }

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step
                wandb.log(log_data, step=step)

            batch_time_m.reset()
            data_time_m.reset()


def evaluate_classification(model, data, epoch, args, num_classes=None):
    """Evaluate classification model on validation data."""
    metrics = {}
    if not is_master(args):
        return metrics

    device = torch.device(args.device)
    model.eval()

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    enabled_modalities = args.enabled_modalities

    if 'val' not in data:
        return metrics

    if not _is_val_epoch(epoch, args.val_frequency, args.epochs):
        return metrics

    eval_model = model
    if args.distributed:
        eval_model = unwrap_model(model)

    if num_classes is None:
        num_classes = eval_model.num_classes

    dataloader = data['val'].dataloader

    all_logits = []
    all_labels = []

    with torch.inference_mode():
        for batch in dataloader:
            batch_tensors = _extract_batch_tensors(batch, enabled_modalities, device, input_dtype)
            labels = _extract_labels(batch, device)

            with autocast():
                logits = eval_model(**batch_tensors)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        num_samples = len(all_labels)

        label_smoothing = getattr(args, 'label_smoothing', 0.0)
        metrics["val_loss"] = F.cross_entropy(
            all_logits, all_labels, label_smoothing=label_smoothing,
        ).item()
        metrics.update(compute_classification_metrics(
            all_logits, all_labels, num_classes=num_classes,
        ))
        metrics["epoch"] = epoch
        metrics["num_samples"] = num_samples

    logging.info(
        f"Eval Epoch: {epoch}  "
        f"loss: {metrics['val_loss']:.4f}  "
        f"acc: {metrics.get('accuracy', 0):.4f}  "
        f"F1: {metrics.get('macro_f1', 0):.4f}"
    )

    if args.save_logs:
        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        log_data = {"val/" + name: val for name, val in metrics.items()}
        if 'train' in data:
            dataloader = data['train'].dataloader
            wandb_step = dataloader.num_batches * epoch
        else:
            wandb_step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=wandb_step)

    return metrics
