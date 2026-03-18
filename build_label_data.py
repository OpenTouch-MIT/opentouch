"""Convert labeled glove HDF5 recordings into a Hugging Face dataset for classification.

HDF5 structure (per file):
    data/
        <clip_id>/
            rgb_images_jpeg       -- JPEG-encoded RGB frames
            right_pressure        -- 16x16 tactile pressure grids
            right_hand_landmarks  -- 21x3 hand joint coordinates
            timestamps            -- per-frame timestamps

Usage::
    python build_label_data.py \
        --input-dir data \
        --output-dir preprocessed_data/classification_peak \
        --label-mapping-path final_annotations \
        --label-column action \
        --frame-index-column peak_idx \
        --temporal-radius 10
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import functools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
from datasets import Array2D, Dataset, Features, Image, Value
from tqdm import tqdm


# ── Configuration ────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class Config:
    image_size: Optional[Tuple[int, int]]
    pressure_max: float
    label_key: str          # "scene_clip" | "scene::clip" | "clip_id"
    label_column: str
    index_column: str
    temporal_before: int
    temporal_after: int
    metadata_columns: Sequence[str]


DEFAULT_METADATA = ("object_name", "object_category", "environment", "action", "grip_type")


# ── Label loading ────────────────────────────────────────────────────

def _load_csv(path: Path, key_col: Optional[str], label_col: str) -> Dict[str, Dict[str, str]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if key_col is None:
            for c in ("key", "scene_clip", "clip_id", "clip", "clip_name"):
                if c in (reader.fieldnames or []):
                    key_col = c
                    break
        if key_col is None:
            raise ValueError(f"Cannot detect key column in {path}. Use --label-key-column.")
        rows = {}
        for row in reader:
            k = (row.get(key_col) or "").strip()
            if k:
                rows[k] = {c: (v or "").strip() for c, v in row.items()}
        return rows


def load_labels(path: str, key_col: Optional[str], label_col: str) -> Dict[str, Dict[str, str]]:
    """Load label rows from a CSV file or directory of CSVs."""
    p = Path(path)
    files = sorted(p.rglob("*.[ct]sv")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"No label files found: {path}")
    combined = {}
    for f in files:
        combined.update(_load_csv(f, key_col, label_col))
    if not combined:
        raise ValueError(f"No label entries found in: {path}")
    return combined


# ── HDF5 helpers ─────────────────────────────────────────────────────

def _build_clip_key(scene: str, clip_id: str, mode: str) -> str:
    if mode == "scene_clip":
        return f"{scene}_{clip_id}"
    if mode == "scene::clip":
        return f"{scene}::{clip_id}"
    return clip_id


def _window_indices(num_frames: int, center: int, before: int, after: int) -> list[int]:
    center = max(0, min(num_frames - 1, center))
    return list(range(max(0, center - before), min(num_frames, center + after + 1)))


def _decode_rgb(jpeg_bytes: np.ndarray, size: Optional[Tuple[int, int]]) -> Optional[np.ndarray]:
    bgr = cv2.imdecode(np.frombuffer(jpeg_bytes.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if size and (rgb.shape[1], rgb.shape[0]) != size:
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    return rgb


# ── Frame processing ─────────────────────────────────────────────────

def _process_frame(frame_args: Tuple, cfg: Config) -> Optional[Dict[str, Any]]:
    scene, clip_id, idx, label_idx, rgb_raw, pressure, landmarks, extra = frame_args

    decoded = _decode_rgb(rgb_raw, cfg.image_size)
    if decoded is None:
        return None

    pr = np.asarray(pressure, dtype=np.float32)
    pr = np.where(np.isfinite(pr), pr, 0.0)
    pr_img = (np.clip(pr, 0, cfg.pressure_max) / cfg.pressure_max * 255).astype(np.uint8)

    lm = np.nan_to_num(landmarks.astype(np.float32, copy=False), nan=0.0)

    record = {
        "scene": scene,
        "clip_id": clip_id,
        "frame_idx": idx,
        "action_label": label_idx,
        "rgb_image": decoded,
        "right_pressure_image": pr_img,
        "right_hand_landmarks": lm,
    }
    record.update(extra)
    return record


def iter_frames(
    file_paths: Sequence[str],
    label_rows: Dict[str, Dict[str, str]],
    label_to_idx: Dict[str, int],
    cfg: Config,
    num_workers: int,
    batch_size: int,
) -> Iterator[Dict[str, Any]]:
    """Yield processed frame records from all HDF5 files."""
    process_fn = functools.partial(_process_frame, cfg=cfg)
    pool = ThreadPoolExecutor(max_workers=num_workers) if num_workers > 0 else None
    map_fn = pool.map if pool else map

    try:
        for path in file_paths:
            scene = Path(path).stem
            with h5py.File(path, "r") as hdf:
                data_group = hdf.get("data")
                if data_group is None:
                    continue
                for clip_id, clip in data_group.items():
                    key = _build_clip_key(scene, clip_id, cfg.label_key)
                    row = label_rows.get(key)
                    if row is None:
                        continue

                    label_val = row.get(cfg.label_column, "").strip()
                    if not label_val or label_val not in label_to_idx:
                        continue

                    ds = {k: clip.get(k) for k in ("timestamps", "rgb_images_jpeg", "right_pressure", "right_hand_landmarks")}
                    if any(v is None for v in ds.values()):
                        continue

                    raw_idx = row.get(cfg.index_column, "").strip()
                    if not raw_idx:
                        continue
                    try:
                        target_idx = int(float(raw_idx)) if "." in raw_idx else int(raw_idx)
                    except ValueError:
                        continue

                    nf = min(d.shape[0] for d in ds.values())
                    if nf <= 0 or target_idx < 0 or target_idx >= nf:
                        continue

                    indices = _window_indices(nf, target_idx, cfg.temporal_before, cfg.temporal_after)
                    if not indices:
                        continue

                    rgb_all = ds["rgb_images_jpeg"][()]
                    pr_all = ds["right_pressure"][()].astype(np.float32, copy=False)
                    lm_all = ds["right_hand_landmarks"][()].astype(np.float32, copy=False)
                    ts_all = np.asarray(ds["timestamps"][()], dtype=np.int64)
                    target_ts = int(ts_all[target_idx])
                    label_idx = label_to_idx[label_val]

                    # Build frame tuples for this clip
                    batch_buf = []
                    for idx in indices:
                        extra: Dict[str, Any] = {
                            "action_label_text": label_val,
                            "target_frame_idx": target_idx,
                            "target_timestamp": target_ts,
                            "frame_timestamp": int(ts_all[idx]),
                            "frame_offset": idx - target_idx,
                        }
                        for col in cfg.metadata_columns:
                            if col not in extra:
                                extra[col] = row.get(col, "")

                        batch_buf.append((scene, clip_id, idx, label_idx, rgb_all[idx], pr_all[idx], lm_all[idx], extra))

                        if len(batch_buf) >= batch_size:
                            for rec in map_fn(process_fn, batch_buf):
                                if rec is not None:
                                    yield rec
                            batch_buf.clear()

                    if batch_buf:
                        for rec in map_fn(process_fn, batch_buf):
                            if rec is not None:
                                yield rec
    finally:
        if pool is not None:
            pool.shutdown(wait=True)


# ── Dataset building ─────────────────────────────────────────────────

def build_features(metadata_columns: Sequence[str]) -> Features:
    feats = {
        "scene": Value("string"),
        "clip_id": Value("string"),
        "frame_idx": Value("int32"),
        "action_label": Value("int32"),
        "rgb_image": Image(),
        "right_pressure_image": Array2D(shape=(16, 16), dtype="uint8"),
        "right_hand_landmarks": Array2D(shape=(21, 3), dtype="float32"),
        "action_label_text": Value("string"),
        "target_frame_idx": Value("int32"),
        "target_timestamp": Value("int64"),
        "frame_timestamp": Value("int64"),
        "frame_offset": Value("int32"),
    }
    for col in metadata_columns:
        if col not in feats:
            feats[col] = Value("string")
    return Features(feats)


def main() -> None:
    p = argparse.ArgumentParser(description="Build labeled HF dataset for classification.")
    p.add_argument("--input-dir", default="data")
    p.add_argument("--output-dir", default="preprocessed_data/classification_dataset")
    p.add_argument("--label-mapping-path", required=True)
    p.add_argument("--label-key", default="scene::clip", choices=["scene_clip", "scene::clip", "clip_id"])
    p.add_argument("--label-key-column", default=None)
    p.add_argument("--label-column", default="action")
    p.add_argument("--frame-index-column", required=True)
    p.add_argument("--temporal-radius", type=int, default=10)
    p.add_argument("--temporal-before", type=int, default=None)
    p.add_argument("--temporal-after", type=int, default=None)
    p.add_argument("--include-label-columns", type=str, nargs="*", default=None)
    p.add_argument("--image-size", type=int, nargs=2, metavar=("W", "H"), default=[224, 224])
    p.add_argument("--pressure-max", type=float, default=3072.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--process-batch-size", type=int, default=512)
    args = p.parse_args()

    # Load labels
    label_rows = load_labels(args.label_mapping_path, args.label_key_column, args.label_column)

    # Resolve metadata columns
    available = sorted({c for row in label_rows.values() for c in row})
    if args.include_label_columns:
        meta_cols = [c for c in args.include_label_columns if c in available]
    else:
        meta_cols = [c for c in DEFAULT_METADATA if c in available]

    # Label index
    labels = sorted({row.get(args.label_column, "").strip() for row in label_rows.values()} - {""})
    if not labels:
        raise ValueError(f"No values found in column '{args.label_column}'.")
    label_to_idx = {l: i for i, l in enumerate(labels)}
    idx_to_label = {i: l for l, i in label_to_idx.items()}
    print(f"Labels ({len(labels)}): {labels}")

    # Temporal window
    t_before = args.temporal_before if args.temporal_before is not None else args.temporal_radius
    t_after = args.temporal_after if args.temporal_after is not None else args.temporal_radius

    cfg = Config(
        image_size=tuple(args.image_size) if args.image_size else None,
        pressure_max=args.pressure_max,
        label_key=args.label_key,
        label_column=args.label_column,
        index_column=args.frame_index_column,
        temporal_before=max(0, t_before),
        temporal_after=max(0, t_after),
        metadata_columns=meta_cols,
    )

    # Build dataset
    file_paths = sorted(os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith(".hdf5"))
    if not file_paths:
        raise FileNotFoundError(f"No .hdf5 files in {args.input_dir}")

    def gen():
        yield from tqdm(
            iter_frames(file_paths, label_rows, label_to_idx, cfg, args.num_workers, args.process_batch_size),
            desc="Extracting frames",
            unit="frame",
        )

    dataset = Dataset.from_generator(gen, features=build_features(meta_cols))
    dataset = dataset.shuffle(seed=args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(args.output_dir)
    with open(os.path.join(args.output_dir, "label_info.json"), "w") as f:
        json.dump(idx_to_label, f, indent=2)
    print(f"Saved {len(dataset)} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
