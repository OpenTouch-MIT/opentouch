"""Data loading for OpenTouch cross-modal retrieval training."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from datasets import Dataset as HFDataset, DatasetDict, load_from_disk
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from opentouch.constants import IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)


VALID_MODALITIES = {"visual", "tactile", "pose"}

TASK_ALIASES = {
    "v2t": (["visual"], ["tactile"]),
    "p2t": (["pose"], ["tactile"]),
    "v2p": (["visual"], ["pose"]),
    "vp2t": (["visual", "pose"], ["tactile"]),
    "tp2v": (["tactile", "pose"], ["visual"]),
    "vt2p": (["visual", "tactile"], ["pose"]),
}


def parse_task(task_str: str) -> Tuple[List[str], List[str]]:
    """Parse task alias (e.g. 'v2t') into (query_mods, target_mods)."""
    key = task_str.strip().lower()
    if key not in TASK_ALIASES:
        raise ValueError("Invalid modality input.")
    return TASK_ALIASES[key]

MODALITY_TO_BATCH_KEY = {
    "visual": "rgb_images",
    "tactile": "tactile_pressure",
    "pose": "hand_landmarks",
}

MODALITY_TO_FEATURE_KEY = {
    "visual": "visual_features",
    "tactile": "tactile_features",
    "pose": "pose_features",
}


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None

    def set_epoch(self, epoch) -> None:
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def group_frames_by_video_clip(dataset: HFDataset) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
    """Return (scene, clip_id) -> sorted list of (frame_idx, row_idx)."""
    clip_to_frames: Dict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)
    for row_idx, scene_name, clip_id, frame_idx in zip(
        range(len(dataset["scene"])),
        dataset["scene"],
        dataset["clip_id"],
        dataset["frame_idx"],
        strict=True,
    ):
        clip_to_frames[(scene_name, clip_id)].append((int(frame_idx), row_idx))
    for frame_list in clip_to_frames.values():
        frame_list.sort(key=lambda item: item[0])
    return clip_to_frames


def split_clips_into_train_val_test(
    clip_keys: Sequence[Tuple[str, str]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Tuple[str, str]]]:
    clips = list(clip_keys)
    if not clips:
        return {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    rng.shuffle(clips)
    n = len(clips)
    n_test = min(int(round(n * test_ratio)), n)
    n_val = min(int(round(n * val_ratio)), max(0, n - n_test))
    n_train = max(0, n - n_val - n_test)
    if n_train == 0:
        n_train = 1
        if n_val > n_test:
            n_val = max(0, n_val - 1)
        elif n_test > 0:
            n_test -= 1
    return {
        "train": clips[:n_train],
        "val": clips[n_train : n_train + n_val],
        "test": clips[n_train + n_val : n_train + n_val + n_test],
    }


class VideoTactilePoseDataset(Dataset):
    """Sliding-window dataset over HuggingFace video-tactile-pose data."""

    def __init__(
        self,
        hf_dataset_path: str,
        sequence_length: int = 20,
        stride: Optional[int] = None,
        split: str = "train",
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        include_tactile: bool = True,
        include_visual: bool = True,
        include_pose: bool = True,
        image_size: Optional[Tuple[int, int]] = None,
        image_transform: Optional[T.Compose] = None,
        _preloaded: Optional[HFDataset] = None,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.stride = stride or sequence_length
        self.include_tactile = include_tactile
        self.include_visual = include_visual
        self.include_pose = include_pose

        if _preloaded is not None:
            selected_dataset = _preloaded
        else:
            loaded_dataset = load_from_disk(hf_dataset_path)
            if isinstance(loaded_dataset, DatasetDict):
                selected_dataset = loaded_dataset[split]
            else:
                full_dataset: HFDataset = loaded_dataset
                clip_to_frames = group_frames_by_video_clip(full_dataset)
                split_to_clips = split_clips_into_train_val_test(
                    clip_keys=list(clip_to_frames.keys()),
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    seed=random_seed,
                )
                clips_for_split = split_to_clips.get(split, [])
                dataset_indices: List[int] = []
                for clip_key in clips_for_split:
                    frame_records = clip_to_frames.get(clip_key, [])
                    dataset_indices.extend(row_idx for _, row_idx in frame_records)
                selected_dataset = full_dataset.select(dataset_indices)

        self.dataset: HFDataset = selected_dataset
        self.clip_groups = group_frames_by_video_clip(self.dataset)

        if image_transform is not None:
            self.image_transform = image_transform
        else:
            transforms = [
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
            if image_size is not None:
                transforms.insert(0, T.Resize(image_size))
            self.image_transform = T.Compose(transforms)

        self.windows: List[Tuple[int, int]] = []
        self.clip_keys: List[Tuple[str, str]] = []
        self.clip_row_indices: List[np.ndarray] = []
        self.clip_frame_indices: List[np.ndarray] = []
        self._build_sliding_windows()

    def _build_sliding_windows(self) -> None:
        for (scene_name, clip_id), frame_records in self.clip_groups.items():
            n_frames = len(frame_records)
            if n_frames < self.sequence_length:
                continue
            frame_nums = np.array([f for f, _ in frame_records], dtype=np.int64)
            row_indices = np.array([r for _, r in frame_records], dtype=np.int64)
            clip_idx = len(self.clip_keys)
            self.clip_keys.append((scene_name, clip_id))
            self.clip_frame_indices.append(frame_nums)
            self.clip_row_indices.append(row_indices)
            for start in range(0, n_frames - self.sequence_length + 1, self.stride):
                self.windows.append((clip_idx, start))
        self.clip_groups.clear()
        if not self.windows:
            raise ValueError("No valid sliding windows. Reduce sequence_length or check dataset.")
        logger.info("Built %d sliding windows from %d clips.", len(self.windows), len(self.clip_keys))

    def __len__(self) -> int:
        return len(self.windows)

    def load_rgb_frames(self, images: Sequence) -> torch.Tensor:
        transformed_frames: List[torch.Tensor] = []
        for image in images:
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image.convert("RGB")
            transformed_frames.append(self.image_transform(pil_image))
        return torch.stack(transformed_frames, dim=0)

    @staticmethod
    def load_tactile_pressure(pressure_images: Sequence[np.ndarray]) -> torch.Tensor:
        pressure = np.stack(pressure_images, axis=0).astype(np.float32)
        return torch.from_numpy(pressure).unsqueeze(1) / 255.0

    @staticmethod
    def load_hand_landmarks(landmarks: Sequence[np.ndarray]) -> torch.Tensor:
        stacked = np.stack(landmarks, axis=0).astype(np.float32)
        return torch.from_numpy(stacked).unsqueeze(1)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        clip_idx, start = self.windows[idx]
        scene, clip_id = self.clip_keys[clip_idx]
        end = start + self.sequence_length
        row_idxs = self.clip_row_indices[clip_idx][start:end].tolist()
        frame_idxs = self.clip_frame_indices[clip_idx][start:end]
        data = self.dataset[row_idxs]

        sample: Dict[str, Any] = {
            "scene": scene,
            "clip_id": clip_id,
            "frame_indices": torch.tensor(frame_idxs, dtype=torch.long),
            "demo_key": f"{scene}_{clip_id}",
            "window_idx": idx,
        }

        if self.include_visual:
            sample["rgb_images"] = self.load_rgb_frames(data["rgb_image"])
        if self.include_tactile:
            sample["tactile_pressure"] = self.load_tactile_pressure(data["right_pressure_image"])
        if self.include_pose:
            sample["hand_landmarks"] = self.load_hand_landmarks(data["right_hand_landmarks"])

        return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        return {}
    result: Dict[str, Any] = {
        "demo_keys": [s.pop("demo_key") for s in batch],
        "window_indices": torch.tensor([s.pop("window_idx") for s in batch], dtype=torch.long),
    }
    for key in batch[0].keys():
        values = [s[key] for s in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values, dim=0)
        else:
            result[key] = values
    return result


def _determine_modality_flags(task_type: str) -> Dict[str, bool]:
    query_mods, target_mods = parse_task(task_type)
    all_mods = set(query_mods) | set(target_mods)
    return {
        "include_visual": "visual" in all_mods,
        "include_tactile": "tactile" in all_mods,
        "include_pose": "pose" in all_mods,
    }


def _load_and_split_dataset(dataset_path, val_ratio, test_ratio, seed):
    """Load dataset once and split into train/val/test subsets."""
    loaded_dataset = load_from_disk(dataset_path)
    if isinstance(loaded_dataset, DatasetDict):
        return dict(loaded_dataset)

    full_dataset: HFDataset = loaded_dataset
    clip_to_frames = group_frames_by_video_clip(full_dataset)
    split_to_clips = split_clips_into_train_val_test(
        clip_keys=list(clip_to_frames.keys()),
        val_ratio=val_ratio, test_ratio=test_ratio, seed=seed,
    )
    splits = {}
    for split_name, clip_keys in split_to_clips.items():
        indices: List[int] = []
        for clip_key in clip_keys:
            frame_records = clip_to_frames.get(clip_key, [])
            indices.extend(row_idx for _, row_idx in frame_records)
        if indices:
            splits[split_name] = full_dataset.select(indices)
    return splits


def get_data(args, epoch=0):
    data = {}

    dataset_path = args.train_data
    if dataset_path is None:
        return data

    task_type = getattr(args, 'task_type', 'v2t')
    modality_flags = _determine_modality_flags(task_type)

    seq_len = getattr(args, 'sequence_length', 20)
    stride = getattr(args, 'stride', None) or seq_len
    val_ratio = getattr(args, 'val_ratio', 0.1)
    test_ratio = getattr(args, 'test_ratio', 0.1)
    seed = getattr(args, 'seed', 42)
    image_size = tuple(getattr(args, 'image_size', (224, 224)))

    common_kwargs = dict(
        sequence_length=seq_len, stride=stride,
        image_size=image_size, **modality_flags,
    )
    val_path = getattr(args, 'val_data', None) or dataset_path
    same_source = val_path == dataset_path

    if same_source:
        splits = _load_and_split_dataset(dataset_path, val_ratio, test_ratio, seed)
        train_preloaded = splits.get("train")
        val_preloaded = splits.get("val")
    else:
        train_splits = _load_and_split_dataset(dataset_path, val_ratio, test_ratio, seed)
        train_preloaded = train_splits.get("train")
        val_splits = _load_and_split_dataset(val_path, val_ratio, test_ratio, seed)
        val_preloaded = val_splits.get("val")

    train_dataset = VideoTactilePoseDataset(
        hf_dataset_path=dataset_path, split="train",
        _preloaded=train_preloaded, **common_kwargs,
    )

    train_sampler = None
    if getattr(args, 'distributed', False):
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True,
        )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), num_workers=args.workers,
        pin_memory=True, sampler=train_sampler,
        drop_last=True, collate_fn=collate_fn,
        persistent_workers=args.workers > 0,
    )
    train_dataloader.num_samples = len(train_dataset)
    train_dataloader.num_batches = len(train_dataloader)

    data["train"] = DataInfo(dataloader=train_dataloader, sampler=train_sampler)

    if val_preloaded is not None and len(val_preloaded) > 0:
        val_dataset = VideoTactilePoseDataset(
            hf_dataset_path=val_path, split="val",
            _preloaded=val_preloaded, **common_kwargs,
        )

        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers,
            pin_memory=True, drop_last=False,
            collate_fn=collate_fn, persistent_workers=args.workers > 0,
        )
        val_dataloader.num_samples = len(val_dataset)
        val_dataloader.num_batches = len(val_dataloader)

        data["val"] = DataInfo(dataloader=val_dataloader)

    return data
