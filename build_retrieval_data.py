"""Convert glove HDF5 recordings into a Hugging Face dataset.

HDF5 structure (per file):
    data/
        <clip_id>/
            rgb_images_jpeg       -- JPEG-encoded RGB frames
            right_pressure        -- 16x16 tactile pressure grids
            right_hand_landmarks  -- 21x3 hand joint coordinates
            timestamps            -- per-frame timestamps

Usage examples::
    # Basic usage with defaults, check arguments with --help
    python build_retrieval_data.py --input-dir data --output-dir preprocessed_data/train_dataset
"""

from __future__ import annotations
import argparse
from curses import raw
import dataclasses
import functools
import os
import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Generator, Iterable, Iterator, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
from datasets import Array2D, Dataset, Features, Image, Value
from tqdm import tqdm
from preprocess.load_data import process_pressure_discrete_intervals


@dataclasses.dataclass(frozen=True)
class PreprocessConfig:
    """Immutable bundle of per-frame processing parameters."""
    image_size: Optional[Tuple[int, int]]  # (width, height) or None to keep original
    pressure_max: float                    # clip ceiling for raw tactile values
    pressure_intervals: int                # number of discrete intensity levels (3, 5, or 7)
    pressure_method: str                   # discretization strategy: "binning", "linear", or "log"
    drop_invalid_poses: bool               # True = discard frames with NaN landmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess glove datasets into a Hugging Face dataset.")
    parser.add_argument("--input-dir", type=str, default="data", help="Directory containing input .hdf5 files.")
    parser.add_argument("--output-dir", type=str, default="preprocessed_data/train_dataset", help="Destination directory for the saved Hugging Face dataset.")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=[224, 224], help="Optional target size (width height) for RGB frames.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when shuffling the dataset.")
    parser.add_argument("--pressure-max", type=float, default=3072.0, help="Maximum expected tactile value before clipping.")
    parser.add_argument("--pressure-intervals", type=int, default=5, choices=[3, 5, 7], help="Number of discrete tactile intensity intervals (3, 5, or 7). Ignored when method is 'none'.")
    parser.add_argument("--pressure-method", type=str, default="none", choices=["none", "binning", "linear", "log"], help="Pressure processing method: none (normalize only), binning, linear, or log.")
    parser.add_argument("--drop-invalid-poses", action="store_true", help="Drop frames whose landmarks contain NaNs instead of zero-filling.")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of worker threads for frame preprocessing. 0 disables threading.")
    parser.add_argument("--process-batch-size", type=int, default=512, help="Number of frames processed together before yielding results.")
    return parser.parse_args()


def list_hdf5_files(input_dir: str) -> Sequence[str]:
    """Return sorted list of .hdf5 file paths in *input_dir*."""
    files = sorted(os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".hdf5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in directory: {input_dir}")
    return files


def count_total_frames(file_paths: Iterable[str]) -> int:
    """Sum frame counts across all HDF5 files (used for the progress bar)."""
    total = 0
    for path in file_paths:
        with h5py.File(path, "r") as hdf:
            data_group = hdf.get("data")
            if data_group is None:
                continue
            for clip in data_group.values():
                if "timestamps" in clip:
                    total += clip["timestamps"].shape[0]
    return total


def resize_rgb(image: np.ndarray, size: Optional[Tuple[int, int]]) -> np.ndarray:
    """Resize *image* to (width, height); no-op when *size* is None or already matches."""
    if size is None or (image.shape[1], image.shape[0]) == size:
        return image
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def decode_rgb(jpeg_bytes: np.ndarray) -> Optional[np.ndarray]:
    """Decode JPEG bytes stored in HDF5 into an RGB uint8 array."""
    bgr = cv2.imdecode(np.frombuffer(jpeg_bytes.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None


def prepare_frame_record(
    frame_args: Tuple[str, str, int, np.ndarray, np.ndarray, np.ndarray],
    config: PreprocessConfig,
) -> Optional[Dict[str, np.ndarray]]:
    """Process a single frame into a dataset record.

    Returns None if the JPEG cannot be decoded or if the frame contains
    invalid landmarks and ``config.drop_invalid_poses`` is set.
    Called via ``functools.partial`` from worker threads in :func:`iter_frames`.
    """
    scene_name, clip_id, frame_idx, rgb_raw, pressure_frame, landmarks_frame = frame_args

    decoded = decode_rgb(rgb_raw)
    if decoded is None:
        return None

    # Discretize 16x16 pressure grid into uint8 intensity levels
    pressure_image = process_pressure_discrete_intervals(
        pressure_frame,
        max_value=config.pressure_max,
        num_intervals=config.pressure_intervals,
        output_range="uint8",
        method=config.pressure_method,
    )

    # Replace NaN landmarks with zeros; optionally drop the frame entirely
    landmarks = landmarks_frame.astype(np.float32, copy=False)
    if config.drop_invalid_poses and np.isnan(landmarks).any():
        return None

    return {
        "scene": scene_name,
        "clip_id": clip_id,
        "frame_idx": frame_idx,
        "rgb_image": resize_rgb(decoded, config.image_size),
        "right_pressure_image": pressure_image,
        "right_hand_landmarks": np.nan_to_num(landmarks, nan=0.0),
    }


def iter_frames(
    file_paths: Sequence[str],
    config: PreprocessConfig,
    num_workers: int,
    process_batch_size: int,
) -> Iterator[Dict[str, np.ndarray]]:
    """Yield processed frame records from all HDF5 files."""
    process_fn = functools.partial(prepare_frame_record, config=config)
    pool = ThreadPoolExecutor(max_workers=num_workers) if num_workers > 0 else None
    map_fn = pool.map if pool else map
    
    try:
        for path in file_paths:
            scene_name = os.path.splitext(os.path.basename(path))[0]
            with h5py.File(path, "r") as hdf:
                data_group = hdf.get("data")
                if data_group is None:
                    continue
                for clip_id, clip_group in data_group.items():
                    # Eagerly load entire clip so the HDF5 handle is not shared across threads
                    rgb_bytes = clip_group["rgb_images_jpeg"][()]
                    pressure_seq = clip_group["right_pressure"][()].astype(np.float32, copy=False)
                    landmarks_seq = clip_group["right_hand_landmarks"][()].astype(np.float32, copy=False)
                    num_frames = landmarks_seq.shape[0]

                    frame_tuples = (
                        (scene_name, clip_id, i, rgb_bytes[i], pressure_seq[i], landmarks_seq[i])
                        for i in range(num_frames)
                    )
                    # Process in batches to bound memory
                    while batch := list(itertools.islice(frame_tuples, process_batch_size)):
                        for record in map_fn(process_fn, batch):
                            if record is not None:
                                yield record
    finally:
        if pool is not None:
            pool.shutdown(wait=True)


def frame_stream(
    file_paths: Sequence[str],
    total_frames: int,
    config: PreprocessConfig,
    num_workers: int,
    process_batch_size: int,
) -> Generator[dict, None, None]:
    """Wrap :func:`iter_frames` with a tqdm progress bar (entry point for ``Dataset.from_generator``)."""
    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as progress:
        for record in iter_frames(file_paths, config, num_workers, process_batch_size):
            progress.update(1)
            yield record


FEATURES = Features({
    "scene": Value("string"),
    "clip_id": Value("string"),
    "frame_idx": Value("int32"),
    "rgb_image": Image(),
    "right_pressure_image": Array2D(shape=(16, 16), dtype="uint8"),
    "right_hand_landmarks": Array2D(shape=(21, 3), dtype="float32"),
})


def main() -> None:
    args = parse_args()
    file_paths = list_hdf5_files(args.input_dir)
    total_frames = count_total_frames(file_paths)
    if total_frames == 0:
        print("No frames found across the provided HDF5 files. Nothing to export.")
        return

    image_size = tuple(args.image_size) if args.image_size else None
    config = PreprocessConfig(
        image_size=image_size,
        pressure_max=args.pressure_max,
        pressure_intervals=args.pressure_intervals,
        pressure_method=args.pressure_method,
        drop_invalid_poses=args.drop_invalid_poses,
    )

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 1) // 2)
    elif num_workers < 0:
        print("num_workers < 0 detected; using single-threaded mode.")
        num_workers = 0

    dataset = Dataset.from_generator(
        frame_stream,
        gen_kwargs={
            "file_paths": file_paths,
            "total_frames": total_frames,
            "config": config,
            "num_workers": num_workers,
            "process_batch_size": max(1, args.process_batch_size),
        },
        features=FEATURES,
    )

    dataset = dataset.shuffle(seed=args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(args.output_dir)
    print(f"Saved Hugging Face dataset to: {args.output_dir}")


if __name__ == "__main__":
    main()
