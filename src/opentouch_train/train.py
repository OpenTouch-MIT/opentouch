"""Training loop for OpenTouch cross-modal retrieval."""

import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from opentouch import get_input_dtype
from opentouch.constants import LOGIT_SCALE_MIN, LOGIT_SCALE_MAX
from opentouch.metrics import compute_retrieval_metrics
from opentouch_train.data import (
    parse_task,
    MODALITY_TO_BATCH_KEY,
    MODALITY_TO_FEATURE_KEY,
)
from opentouch_train.distributed import is_master
from opentouch_train.precision import get_autocast


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def _extract_batch_tensors(batch, modality_names, device, input_dtype):
    """Move modality tensors from batch dict to device."""
    tensors = {}
    kwargs = dict(device=device, non_blocking=True)
    if input_dtype is not None:
        kwargs["dtype"] = input_dtype
    for mod in modality_names:
        batch_key = MODALITY_TO_BATCH_KEY[mod]
        if batch_key in batch:
            tensors[batch_key] = batch[batch_key].to(**kwargs)
    return tensors


def _get_query_target_features(model_out, query_mods, target_mods, model):
    """Extract query/target features from model output, fusing if multi-modal query."""
    if len(target_mods) == 1:
        target_features = model_out[MODALITY_TO_FEATURE_KEY[target_mods[0]]]
    else:
        raise ValueError(f"Multiple target modalities not supported: {target_mods}")

    if len(query_mods) == 1:
        query_features = model_out[MODALITY_TO_FEATURE_KEY[query_mods[0]]]
    else:
        raw_model = unwrap_model(model)
        encoded = {}
        for mod in query_mods:
            encoded[mod] = model_out[MODALITY_TO_FEATURE_KEY[mod]]
        query_features = raw_model.fuse_encoded_features(encoded, target_mods[0])

    return query_features, target_features


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()

    query_mods, target_mods = parse_task(args.task_type)
    all_mods = list(set(query_mods) | set(target_mods))

    data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_batches, accum_features = [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    pbar = tqdm(
        enumerate(dataloader),
        total=num_batches_per_epoch * args.accum_freq,
        desc=f"Epoch {epoch}",
        disable=not is_master(args),
    )
    for i, batch in pbar:
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler and scheduler is not None:
            scheduler(step)

        batch_tensors = _extract_batch_tensors(batch, all_mods, device, input_dtype)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(**batch_tensors)
                logit_scale = model_out["logit_scale"]

                query_features, target_features = _get_query_target_features(
                    model_out, query_mods, target_mods, model,
                )
                losses_dict = loss(
                    query_features,
                    target_features,
                    logit_scale,
                    logit_bias=model_out.get("logit_bias"),
                    output_dict=True,
                )
                total_loss = sum(losses_dict.values())
                losses_dict["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # Gradient accumulation
            with torch.no_grad():
                with autocast():
                    model_out = model(**batch_tensors)
                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)
                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]
                accum_batches.append(batch_tensors)

            if ((i + 1) % args.accum_freq) > 0:
                continue

            optimizer.zero_grad()
            for j in range(args.accum_freq):
                batch_j = accum_batches[j]
                with autocast():
                    model_out = model(**batch_j)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop(
                        "logit_scale"
                    )
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(
                            accumulated[:j] + [model_out[key]] + accumulated[j + 1:]
                        )

                    merged_out = {**inputs, **inputs_no_accum}
                    query_features, target_features = _get_query_target_features(
                        merged_out, query_mods, target_mods, model,
                    )
                    losses_dict = loss(
                        query_features,
                        target_features,
                        logit_scale,
                        logit_bias=inputs_no_accum.get("logit_bias"),
                        output_dict=True,
                    )
                    del inputs, inputs_no_accum
                    total_loss = sum(losses_dict.values())
                    losses_dict["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0,
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0,
                )
            optimizer.step()

        if args.accum_freq > 1:
            accum_batches, accum_features = [], {}

        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(LOGIT_SCALE_MIN, LOGIT_SCALE_MAX)

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1

        if is_master(args) and (
            i_accum % args.log_every_n_steps == 0
            or batch_count == num_batches_per_epoch
        ):
            batch_size = args.batch_size
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            for key, val in losses_dict.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            samples_per_second = (
                args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            )

            pbar.set_postfix(
                loss=f"{list(losses_m.values())[0].avg:.4f}" if losses_m else "N/A",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                sps=f"{samples_per_second:.0f}",
            )

            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": (
                    args.accum_freq * args.batch_size / batch_time_m.val
                ),
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
            }
            log_data.update({name: val.val for name, val in losses_m.items()})
            log_data = {"train/" + name: val for name, val in log_data.items()}

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step
                wandb.log(log_data, step=step)

            batch_time_m.reset()
            data_time_m.reset()


def evaluate(model, data, epoch, args, tb_writer=None):
    """Evaluate on validation data: compute val loss and retrieval metrics."""
    metrics = {}
    if not is_master(args):
        return metrics

    device = torch.device(args.device)
    model.eval()

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    query_mods, target_mods = parse_task(args.task_type)
    all_mods = list(set(query_mods) | set(target_mods))
    query_label = "+".join(query_mods)
    target_label = "+".join(target_mods)
    query_key = query_label.replace("+", "_")
    target_key = target_label.replace("+", "_")

    if "val" in data and (
        args.val_frequency
        and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)
    ):
        eval_model = model
        if args.distributed:
            eval_model = unwrap_model(model)

        dataloader = data['val'].dataloader

        all_query_features, all_target_features = [], []
        logit_scale_val = None

        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                batch_tensors = _extract_batch_tensors(batch, all_mods, device, input_dtype)
                with autocast():
                    model_out = eval_model(**batch_tensors)
                    query_feat, target_feat = _get_query_target_features(
                        model_out, query_mods, target_mods, eval_model,
                    )
                    all_query_features.append(query_feat.cpu())
                    all_target_features.append(target_feat.cpu())
                    if logit_scale_val is None:
                        logit_scale_val = model_out["logit_scale"].mean().cpu()

            all_query = torch.cat(all_query_features)
            all_target = torch.cat(all_target_features)
            num_samples = len(all_query)

            logits_q2t = logit_scale_val * all_query @ all_target.t()
            labels = torch.arange(num_samples).long()
            val_loss = (
                F.cross_entropy(logits_q2t, labels)
                + F.cross_entropy(logits_q2t.t(), labels)
            ) / 2
            retrieval_metrics = compute_retrieval_metrics(
                all_query,
                all_target,
                top_k=[1, 5, 10],
                query_label=query_label,
                target_label=target_label,
            )

            for direction, values in retrieval_metrics.items():
                for name, value in values.items():
                    metrics[f"{direction}_{name}"] = value

            metrics.update({"val_loss": val_loss.item(), "epoch": epoch, "num_samples": num_samples})

    if not metrics:
        return metrics

    fwd = f"{query_key}_to_{target_key}"
    rev = f"{target_key}_to_{query_key}"

    def _fmt(d):
        return (
            f"R@1/5/10: {metrics.get(f'{d}_recall@1', 0):.4f}/"
            f"{metrics.get(f'{d}_recall@5', 0):.4f}/"
            f"{metrics.get(f'{d}_recall@10', 0):.4f}  "
            f"mAP: {metrics.get(f'{d}_mAP', 0):.4f}"
        )
    logging.info(
        f"Eval Epoch: {epoch}  loss: {metrics.get('val_loss', 0):.4f}\n"
        f"  {query_label}->{target_label}  {_fmt(fwd)}\n"
        f"  {target_label}->{query_label}  {_fmt(rev)}"
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics
