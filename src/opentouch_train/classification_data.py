"""Dataset and data loading for peak-window classification training."""

from __future__ import annotations

import logging
import random
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from datasets import DatasetDict, load_from_disk
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from opentouch.constants import IMAGENET_MEAN, IMAGENET_STD
from opentouch_train.data import DataInfo

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

_TASK_TO_COLUMN = {"action": "action", "grip": "grip_type"}
_VALID_SPLITS = {"train", "val", "test"}


class PeakWindowClassificationDataset(Dataset):
    """Peak-window classification dataset built from HF records."""

    def __init__(
        self,
        hf_dataset_path: str,
        sequence_length: int = 20,
        split: str = "train",
        *,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        include_visual: bool = True,
        include_tactile: bool = True,
        include_pose: bool = True,
        image_size: Optional[Tuple[int, int]] = None,
        task: str = "action",
    ) -> None:
        super().__init__()

        self.sequence_length = int(sequence_length)
        self.include_visual = bool(include_visual)
        self.include_tactile = bool(include_tactile)
        self.include_pose = bool(include_pose)
        self.task = task
        if task not in _TASK_TO_COLUMN:
            raise ValueError(
                f"Unsupported task: {task!r}. Choose from {sorted(_TASK_TO_COLUMN)}"
            )

        raw_dataset = load_from_disk(hf_dataset_path)
        if isinstance(raw_dataset, DatasetDict):
            dataset = raw_dataset["train"]
        else:
            dataset = raw_dataset

        self.dataset = dataset

        self._scenes = dataset["scene"]
        self._clips = dataset["clip_id"]
        self._target_frame_indices = dataset["target_frame_idx"]
        self._frame_indices = dataset["frame_idx"]
        self._frame_offsets = self._get_optional_column(dataset, "frame_offset")
        self._label_column = dataset[_TASK_TO_COLUMN[task]]
        self._target_timestamps = dataset["target_timestamp"]

        self.sequences = self._build_sequences()
        if not self.sequences:
            raise ValueError("No valid sequences constructed from dataset.")

        self.classes, self.class_to_idx = self._build_label_vocab()

        self.samples = self._split_sequences(split, val_ratio, test_ratio, random_seed)
        if not self.samples:
            raise ValueError(f"No samples for split '{split}'. Adjust ratios or seed.")

        self.image_transform = self._build_image_transform(image_size)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @staticmethod
    def _get_optional_column(dataset, column_name):
        return dataset[column_name] if column_name in dataset.column_names else None

    def _build_image_transform(self, image_size):
        if not self.include_visual:
            return None
        transforms: List[Any] = []
        if image_size is not None:
            transforms.append(T.Resize(image_size))
        transforms.append(T.ToTensor())
        transforms.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        return T.Compose(transforms)

    def _build_sequences(self) -> Dict[Tuple, SimpleNamespace]:
        groups: Dict[Tuple, List[int]] = {}
        for idx, (scene, clip, target_idx) in enumerate(
            zip(self._scenes, self._clips, self._target_frame_indices, strict=True)
        ):
            key = (scene, clip, int(target_idx))
            groups.setdefault(key, []).append(idx)

        sequences: Dict[Tuple, SimpleNamespace] = {}
        for key, row_indices in groups.items():
            if self._frame_offsets is not None:
                row_indices.sort(key=lambda i: self._frame_offsets[i])
            else:
                row_indices.sort(key=lambda i: self._frame_indices[i])

            first_idx = row_indices[0]
            label = self._label_column[first_idx] or ""
            if not label:
                continue

            sequences[key] = SimpleNamespace(
                key=key,
                indices=row_indices,
                scene=key[0],
                clip_id=key[1],
                target_frame_idx=int(key[2]),
                label=label,
                target_timestamp=int(self._target_timestamps[first_idx]),
            )

        return sequences

    def _build_label_vocab(self):
        classes = sorted({meta.label for meta in self.sequences.values()})
        class_to_idx = {label: idx for idx, label in enumerate(classes)}
        return classes, class_to_idx

    def _split_sequences(
        self,
        split: str,
        val_ratio: float,
        test_ratio: float,
        random_seed: int,
    ) -> List[SimpleNamespace]:
        split = split.lower()
        clip_to_sequences: Dict[Tuple, List[SimpleNamespace]] = {}
        for meta in self.sequences.values():
            clip_key = (meta.scene, meta.clip_id)
            clip_to_sequences.setdefault(clip_key, []).append(meta)

        train_keys, val_keys, test_keys = self._stratified_split_clips(
            clip_to_sequences, val_ratio, test_ratio, random_seed,
        )

        split_to_keys = {"train": train_keys, "val": val_keys, "test": test_keys}
        selected = split_to_keys[split]
        samples: List[SimpleNamespace] = []
        for clip_key in selected:
            samples.extend(clip_to_sequences[clip_key])
        samples.sort(key=lambda m: (m.scene, m.clip_id, m.target_frame_idx))

        logger.info(
            "Classification split '%s': %d samples from %d clips.",
            split, len(samples), len(selected),
        )
        return samples

    def _stratified_split_clips(
        self,
        clip_to_sequences: Dict[Tuple, List[SimpleNamespace]],
        val_ratio: float,
        test_ratio: float,
        random_seed: int,
    ):
        """Split clips with stratification on the task label."""
        clips_by_class: Dict[str, List[Tuple]] = {}

        for clip_key, sequences in clip_to_sequences.items():
            label = sequences[0].label
            if not label:
                continue
            clips_by_class.setdefault(label, []).append(clip_key)

        return self._stratified_split_sklearn(
            clips_by_class, val_ratio, test_ratio, random_seed,
        )

    @staticmethod
    def _stratified_split_sklearn(
        clips_by_class: Dict[str, List[Tuple]],
        val_ratio: float,
        test_ratio: float,
        random_seed: int,
    ):
        train_keys: List[Tuple] = []
        val_keys: List[Tuple] = []
        test_keys: List[Tuple] = []

        for _label, clips in clips_by_class.items():
            clips_list = list(clips)
            if len(clips_list) < 3:
                if len(clips_list) == 1:
                    train_keys.extend(clips_list)
                else:
                    train_keys.append(clips_list[0])
                    test_keys.append(clips_list[1])
                continue

            train_val, test_part = train_test_split(
                clips_list, test_size=test_ratio,
                random_state=random_seed, shuffle=True,
            )
            adjusted_val = val_ratio / (1 - test_ratio) if (1 - test_ratio) > 0 else val_ratio
            if len(train_val) >= 2:
                train_part, val_part = train_test_split(
                    train_val, test_size=adjusted_val,
                    random_state=random_seed, shuffle=True,
                )
            else:
                train_part, val_part = train_val, []

            train_keys.extend(train_part)
            val_keys.extend(val_part)
            test_keys.extend(test_part)

        rng = random.Random(random_seed)
        rng.shuffle(train_keys)
        rng.shuffle(val_keys)
        rng.shuffle(test_keys)
        return train_keys, val_keys, test_keys

    def _select_indices(self, indices: List[int]):
        """Select up to *sequence_length* indices, padding short sequences."""
        if not indices:
            raise RuntimeError("Encountered empty sequence during indexing.")

        if len(indices) >= self.sequence_length:
            selected = list(indices[: self.sequence_length])
            mask = torch.ones(self.sequence_length, dtype=torch.bool)
            return selected, mask

        pad_value = indices[-1]
        selected = list(indices) + [pad_value] * (self.sequence_length - len(indices))
        mask = torch.zeros(self.sequence_length, dtype=torch.bool)
        mask[: len(indices)] = True
        return selected, mask

    def _load_rgb_sequence(self, row_indices: List[int]) -> torch.Tensor:
        frames: List[torch.Tensor] = []
        for row_idx in row_indices:
            record = self.dataset[row_idx]
            image = record["rgb_image"]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.asarray(image))
            tensor = self.image_transform(image) if self.image_transform else T.ToTensor()(image)
            frames.append(tensor)
        return torch.stack(frames, dim=0)

    def _load_tactile_sequence(self, row_indices: List[int]) -> torch.Tensor:
        frames: List[torch.Tensor] = []
        for row_idx in row_indices:
            array = np.asarray(
                self.dataset[row_idx]["right_pressure_image"], dtype=np.float32,
            )
            if array.ndim == 2:
                array = array[None, ...]
            frames.append(torch.from_numpy(array) / 255.0)
        return torch.stack(frames, dim=0)

    def _load_pose_sequence(self, row_indices: List[int]) -> torch.Tensor:
        frames: List[torch.Tensor] = []
        for row_idx in row_indices:
            landmarks = np.asarray(
                self.dataset[row_idx]["right_hand_landmarks"], dtype=np.float32,
            )
            frames.append(torch.from_numpy(landmarks.reshape(-1, 3)))
        return torch.stack(frames, dim=0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        metadata = self.samples[index]
        selected_indices, mask = self._select_indices(metadata.indices)

        sample: Dict[str, Any] = {
            "scene": metadata.scene,
            "clip_id": metadata.clip_id,
            "target_frame_idx": torch.tensor(metadata.target_frame_idx, dtype=torch.long),
            "sequence_mask": mask,
            "frame_indices": torch.tensor(
                [self._frame_indices[i] for i in selected_indices], dtype=torch.long,
            ),
        }

        if self.include_visual:
            sample["rgb_images"] = self._load_rgb_sequence(selected_indices)
        if self.include_tactile:
            sample["tactile_pressure"] = self._load_tactile_sequence(selected_indices)
        if self.include_pose:
            sample["hand_landmarks"] = self._load_pose_sequence(selected_indices)

        sample["label"] = torch.tensor(
            self.class_to_idx[metadata.label], dtype=torch.long,
        )

        return sample


def collate_classification(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack tensors along a new batch dimension; collect non-tensors as lists."""
    if not batch:
        return {}
    collated: Dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = values
    return collated


def _attach_loader_metadata(loader: DataLoader, dataset: PeakWindowClassificationDataset) -> None:
    loader.num_samples = len(dataset)
    loader.num_batches = len(loader)
    loader.num_classes = dataset.num_classes


def get_classification_data(args) -> Dict[str, DataInfo]:
    """Create train/val DataLoaders for classification."""
    data: Dict[str, DataInfo] = {}
    dataset_path = getattr(args, "train_data", None)
    if dataset_path is None:
        return data

    modalities = set(getattr(args, "enabled_modalities", ("visual", "tactile", "pose")))
    task = getattr(args, "task", "action")
    seq_len = getattr(args, "sequence_length", 20)
    image_size = tuple(getattr(args, "image_size", (224, 224)))
    val_ratio = getattr(args, "val_ratio", 0.1)
    test_ratio = getattr(args, "test_ratio", 0.1)
    seed = getattr(args, "seed", 42)

    common_kwargs = dict(
        hf_dataset_path=dataset_path,
        sequence_length=seq_len,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=seed,
        include_visual="visual" in modalities,
        include_tactile="tactile" in modalities,
        include_pose="pose" in modalities,
        image_size=image_size,
        task=task,
    )

    train_dataset = PeakWindowClassificationDataset(split="train", **common_kwargs)

    train_sampler = None
    if getattr(args, "distributed", False):
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_classification,
        persistent_workers=args.workers > 0,
    )
    _attach_loader_metadata(train_loader, train_dataset)

    data["train"] = DataInfo(dataloader=train_loader, sampler=train_sampler)

    try:
        val_dataset = PeakWindowClassificationDataset(split="val", **common_kwargs)
    except ValueError as e:
        if "No samples for split" in str(e) or "No valid sequences" in str(e):
            logger.warning(f"No validation data available: {e}")
            val_dataset = None
        else:
            raise

    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_classification,
            persistent_workers=args.workers > 0,
        )
        _attach_loader_metadata(val_loader, val_dataset)

        data["val"] = DataInfo(dataloader=val_loader)

    return data
