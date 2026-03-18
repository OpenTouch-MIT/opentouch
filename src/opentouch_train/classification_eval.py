"""Standalone evaluation for classification checkpoints.

All parameters (model, task, modalities, data path, etc.) are auto-detected
from params.txt next to the checkpoint. CLI flags override auto-detected values.

Usage::
    python -m opentouch_train.classification_eval \
        --checkpoint logs/<run>/checkpoints/epoch_<N>.pt
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from opentouch import create_classification_model, get_input_dtype
from opentouch.classification_metrics import compute_classification_metrics
from opentouch_train.classification_data import (
    PeakWindowClassificationDataset, collate_classification,
)
from opentouch_train.classification_train import _extract_batch_tensors, _extract_labels
from opentouch_train.precision import get_autocast


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _parse_params_txt(path):
    """Parse params.txt into a dict, handling list values."""
    params = {}
    for line in Path(path).read_text().splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            params[k.strip()] = v.strip()
    return params


def _parse_list_value(raw):
    """Parse a string like \"['visual', 'tactile']\" into a list."""
    if not raw:
        return None
    return raw.strip("()[] ").replace("'", "").replace('"', '').split(", ")


def _read_checkpoint_meta(path):
    """Read metadata from checkpoint and params.txt (two levels up from checkpoints/)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = {
        "model": ckpt.get("model"),
        "task": ckpt.get("task"),
        "num_classes": ckpt.get("num_classes"),
        "epoch": ckpt.get("epoch"),
    }

    params_file = Path(path).resolve().parent.parent / "params.txt"
    if params_file.exists():
        params = _parse_params_txt(params_file)
        if meta["model"] is None:
            meta["model"] = params.get("model")
        if meta["task"] is None:
            meta["task"] = params.get("task")
        if meta.get("modalities") is None:
            meta["modalities"] = _parse_list_value(params.get("enabled_modalities"))
        # Additional params from params.txt
        for key in ("train_data", "precision", "batch_size", "sequence_length",
                    "val_ratio", "test_ratio", "seed", "workers"):
            if key not in meta or meta.get(key) is None:
                meta[key] = params.get(key)

    return meta


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a classification checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file.")
    p.add_argument("--data", default=None, help="Path to preprocessed HF dataset (auto-detected from params.txt).")
    p.add_argument("--model", default=None, help="Model config name (auto-detected).")
    p.add_argument("--task", default=None, help="Classification task (auto-detected).")
    p.add_argument("--modalities", nargs="+", default=None, help="Modalities (auto-detected).")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--precision", default="amp", choices=["fp32", "amp", "bf16"])
    p.add_argument("--sequence-length", type=int, default=20)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None, help="Optional path to save metrics JSON.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    meta = _read_checkpoint_meta(args.checkpoint)
    if args.model is None:
        args.model = meta.get("model") or "OpenTouch-DINOv3-B16-Classify"
    if args.task is None:
        args.task = meta.get("task") or "action"
    if args.modalities is None:
        args.modalities = meta.get("modalities") or ["visual", "tactile"]
    if args.data is None:
        args.data = meta.get("train_data")
        if args.data is None:
            raise ValueError("Cannot auto-detect data path from params.txt. Please specify --data.")
    # Use params.txt defaults for unset numeric args
    if meta.get("batch_size"):
        args.batch_size = args.batch_size or int(meta["batch_size"])
    if meta.get("sequence_length"):
        args.sequence_length = args.sequence_length or int(meta["sequence_length"])
    if meta.get("precision"):
        args.precision = args.precision or meta["precision"]
    if meta.get("val_ratio"):
        args.val_ratio = args.val_ratio or float(meta["val_ratio"])
    if meta.get("test_ratio"):
        args.test_ratio = args.test_ratio or float(meta["test_ratio"])
    if meta.get("seed"):
        args.seed = args.seed or int(meta["seed"])
    if meta.get("workers"):
        args.workers = args.workers or int(meta["workers"])
    num_classes = meta.get("num_classes")

    log.info(f"Model: {args.model}  Task: {args.task}  Modalities: {args.modalities}  Epoch: {meta.get('epoch', '?')}")

    device = torch.device(args.device)

    # Load dataset to discover num_classes if not in checkpoint
    dataset = PeakWindowClassificationDataset(
        hf_dataset_path=args.data,
        split=args.split,
        sequence_length=args.sequence_length,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        include_visual="visual" in args.modalities,
        include_tactile="tactile" in args.modalities,
        include_pose="pose" in args.modalities,
        image_size=(224, 224),
        task=args.task,
    )
    if num_classes is None:
        num_classes = dataset.num_classes

    model = create_classification_model(
        args.model,
        num_classes=num_classes,
        pretrained=args.checkpoint,
        precision=args.precision,
        device=device,
        enabled_modalities=args.modalities,
    )
    model.eval()

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False,
        collate_fn=collate_classification, persistent_workers=args.workers > 0,
    )
    log.info(f"Split: {args.split}  samples: {len(dataset)}  classes: {num_classes}")

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    all_logits, all_labels = [], []

    with torch.inference_mode():
        for batch in dataloader:
            batch_tensors = _extract_batch_tensors(batch, args.modalities, device, input_dtype)
            labels = _extract_labels(batch, device)

            with autocast():
                logits = model(**batch_tensors)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    num_samples = len(all_labels)

    metrics = {
        "split": args.split,
        "task": args.task,
        "num_samples": num_samples,
    }
    metrics.update(compute_classification_metrics(
        all_logits, all_labels, num_classes=num_classes,
    ))

    class_names = dataset.classes

    log.info(
        f"\n{'='*60}\n"
        f"  Checkpoint : {args.checkpoint}\n"
        f"  Split      : {args.split}  ({num_samples} samples)\n"
        f"  Task       : {args.task}  ({num_classes} classes: {class_names})\n"
        f"\n"
        f"  Accuracy   : {metrics.get('accuracy', 0):.4f}\n"
        f"  Macro F1   : {metrics.get('macro_f1', 0):.4f}\n"
        f"{'='*60}"
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        log.info(f"Saved metrics to {args.output}")

    return metrics


if __name__ == "__main__":
    main()
