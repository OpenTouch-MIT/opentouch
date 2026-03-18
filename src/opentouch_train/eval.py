"""Standalone evaluation for retrieval checkpoints.

Usage::
    python -m opentouch_train.eval \
        --checkpoint logs/multi_gpu_v2t/checkpoints/epoch_30.pt \
        --data preprocessed_data/train_dataset
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from opentouch import create_model, compute_retrieval_metrics, get_input_dtype
from opentouch_train.data import (
    VideoTactilePoseDataset, collate_fn, parse_task, _determine_modality_flags,
)
from opentouch_train.precision import get_autocast


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _read_params_file(params_file: Path) -> dict:
    params = {}
    if not params_file.exists():
        return params
    for line in params_file.read_text().splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            params[key.strip()] = value.strip()
    return params


def _expand_retrieval_metrics(retrieval_metrics: dict) -> dict:
    metrics = {}
    for direction, values in retrieval_metrics.items():
        for name, value in values.items():
            metrics[f"{direction}_{name}"] = value
    return metrics


def _direction_keys(query_mods, target_mods):
    query_label = "+".join(query_mods)
    target_label = "+".join(target_mods)
    query_key = query_label.replace("+", "_")
    target_key = target_label.replace("+", "_")
    return query_label, target_label, f"{query_key}_to_{target_key}", f"{target_key}_to_{query_key}"


def _read_checkpoint_meta(path):
    """Read metadata from checkpoint, falling back to params.txt in the log dir."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = {
        "task_type": ckpt.get("task_type"),
        "model": ckpt.get("model"),
        "epoch": ckpt.get("epoch"),
    }

    if meta["task_type"] is None or meta["model"] is None:
        params_file = Path(path).resolve().parent.parent / "params.txt"
        params = _read_params_file(params_file)
        if meta["task_type"] is None:
            meta["task_type"] = params.get("task_type")
        if meta["model"] is None:
            meta["model"] = params.get("model")

    return meta


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a retrieval checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file.")
    p.add_argument("--data", required=True, help="Path to preprocessed HF dataset.")
    p.add_argument("--model", default=None, help="Model config name (auto-detected from checkpoint).")
    p.add_argument("--task-type", default=None, help="Retrieval task (auto-detected from checkpoint).")
    p.add_argument("--split", default="test", choices=["val", "test"], help="Dataset split.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--precision", default="amp", choices=["fp32", "amp", "bf16"])
    p.add_argument("--sequence-length", type=int, default=20)
    p.add_argument("--val-ratio", type=float, default=0.1, help="Val split ratio (must match training).")
    p.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio (must match training).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split (must match training).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None, help="Optional path to save metrics JSON.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    meta = _read_checkpoint_meta(args.checkpoint)
    if args.task_type is None:
        args.task_type = meta.get("task_type")
        if args.task_type is None:
            raise ValueError(
                "Checkpoint does not contain task_type. "
                "Specify --task-type explicitly (v2t, p2t, v2p, vp2t, …)."
            )
    if args.model is None:
        args.model = meta.get("model") or "OpenTouch-DINOv3-B16-Retrieval"

    log.info(f"Model: {args.model}  Task: {args.task_type}  Epoch: {meta.get('epoch', '?')}")

    device = torch.device(args.device)
    query_mods, target_mods = parse_task(args.task_type)
    modality_flags = _determine_modality_flags(args.task_type)

    model = create_model(args.model, pretrained=args.checkpoint, precision=args.precision, device=device)
    model.eval()

    dataset = VideoTactilePoseDataset(
        hf_dataset_path=args.data,
        split=args.split,
        sequence_length=args.sequence_length,
        image_size=(224, 224),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        **modality_flags,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False,
        collate_fn=collate_fn, persistent_workers=args.workers > 0,
    )
    log.info(f"Split: {args.split}  samples: {len(dataset)}  batches: {len(dataloader)}")

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    all_mods = list(set(query_mods) | set(target_mods))

    all_query_features, all_target_features = [], []
    logit_scale_val = None

    from opentouch_train.train import _extract_batch_tensors, _get_query_target_features

    with torch.inference_mode():
        for batch in dataloader:
            batch_tensors = _extract_batch_tensors(batch, all_mods, device, input_dtype)
            with autocast():
                model_out = model(**batch_tensors)
                query_feat, target_feat = _get_query_target_features(
                    model_out, query_mods, target_mods, model,
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
        all_query, all_target, top_k=[1, 5, 10],
        query_label="+".join(query_mods),
        target_label="+".join(target_mods),
    )

    query_label, target_label, fwd, rev = _direction_keys(query_mods, target_mods)
    metrics = {
        "split": args.split,
        "val_loss": val_loss.item(),
        "num_samples": num_samples,
        **_expand_retrieval_metrics(retrieval_metrics),
    }

    log.info(
        f"\n{'='*60}\n"
        f"  Checkpoint : {args.checkpoint}\n"
        f"  Split      : {args.split}  ({num_samples} samples)\n"
        f"\n"
        f"  {query_label} -> {target_label}\n"
        f"    R@1  : {metrics.get(f'{fwd}_recall@1', 0):.4f}\n"
        f"    R@5  : {metrics.get(f'{fwd}_recall@5', 0):.4f}\n"
        f"    R@10 : {metrics.get(f'{fwd}_recall@10', 0):.4f}\n"
        f"    mAP  : {metrics.get(f'{fwd}_mAP', 0):.4f}\n"
        f"\n"
        f"  {target_label} -> {query_label}\n"
        f"    R@1  : {metrics.get(f'{rev}_recall@1', 0):.4f}\n"
        f"    R@5  : {metrics.get(f'{rev}_recall@5', 0):.4f}\n"
        f"    R@10 : {metrics.get(f'{rev}_recall@10', 0):.4f}\n"
        f"    mAP  : {metrics.get(f'{rev}_mAP', 0):.4f}\n"
        f"{'='*60}"
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        log.info(f"Saved metrics to {args.output}")

    return metrics


if __name__ == "__main__":
    main()
