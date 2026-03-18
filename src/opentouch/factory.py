"""Factory for creating models and loss functions."""

import inspect
import json
import logging
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .classification import ClassificationModel
from .loss import ClipLoss
from .model import CrossRetrievalModel, get_cast_dtype

_MODEL_VALID_PARAMS = set(
    inspect.signature(CrossRetrievalModel.__init__).parameters.keys()
) - {'self'}

_CLASSIFICATION_VALID_PARAMS = set(
    inspect.signature(ClassificationModel.__init__).parameters.keys()
) - {'self'}

_MODEL_CONFIG_PATHS = [Path(__file__).parent / "model_configs/"]
_MODEL_CONFIGS = {}


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS
    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))
    for config_file in config_files:
        with open(config_file, 'r') as f:
            _MODEL_CONFIGS[config_file.stem] = json.load(f)
    _MODEL_CONFIGS.update({k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: natural_key(x[0]))})


_rescan_model_configs()


def list_models():
    return list(_MODEL_CONFIGS.keys())


def add_model_config(model_name: str, config: dict):
    _MODEL_CONFIGS[model_name] = config


def get_model_config(model_name: str) -> Optional[dict]:
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    return None


def create_model(
    model_name: str,
    pretrained: str = '',
    precision: str = 'fp32',
    device: Union[str, torch.device] = 'cpu',
    cache_dir: Optional[str] = None,
    output_dict: Optional[bool] = None,
    **model_kwargs,
) -> CrossRetrievalModel:
    model_cfg = get_model_config(model_name)
    if model_cfg is None:
        raise ValueError(f"Model config '{model_name}' not found. Available: {list_models()}")

    for k, v in model_kwargs.items():
        if k in model_cfg and isinstance(model_cfg[k], dict) and isinstance(v, dict):
            model_cfg[k].update(v)
        else:
            model_cfg[k] = v

    if cache_dir is not None:
        os.environ.setdefault('HF_HOME', cache_dir)

    filtered_cfg = {k: v for k, v in model_cfg.items() if k in _MODEL_VALID_PARAMS}
    model = CrossRetrievalModel(**filtered_cfg)

    if pretrained:
        load_checkpoint(model, pretrained)

    model = model.to(device)
    cast_dtype = get_cast_dtype(precision)
    if cast_dtype is not None:
        model = model.to(dtype=cast_dtype)

    return model


def create_model_and_transforms(
    model_name: str,
    pretrained: str = '',
    precision: str = 'fp32',
    device: Union[str, torch.device] = 'cpu',
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Tuple[CrossRetrievalModel, None, None]:
    """Return (model, None, None); transforms are dataset-defined."""
    model = create_model(
        model_name, pretrained=pretrained, precision=precision,
        device=device, cache_dir=cache_dir, **kwargs,
    )
    return model, None, None


def load_checkpoint(
    model: Union[CrossRetrievalModel, ClassificationModel],
    checkpoint_path: str,
    strict: bool = True,
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    if next(iter(state_dict.keys())).startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    result = model.load_state_dict(state_dict, strict=strict)
    if not strict and (result.missing_keys or result.unexpected_keys):
        logging.info(
            f"Loaded checkpoint from '{checkpoint_path}' with strict=False: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    else:
        logging.info(f"Loaded checkpoint from '{checkpoint_path}'")


def create_loss(args):
    return ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
    )


def create_classification_model(
    model_name: str,
    num_classes: int,
    pretrained: str = '',
    precision: str = 'fp32',
    device: Union[str, torch.device] = 'cpu',
    cache_dir: Optional[str] = None,
    **model_kwargs,
) -> ClassificationModel:
    """Create a ClassificationModel from a named config."""
    model_cfg = get_model_config(model_name)
    if model_cfg is None:
        raise ValueError(f"Model config '{model_name}' not found. Available: {list_models()}")

    for k, v in model_kwargs.items():
        if k in model_cfg and isinstance(model_cfg[k], dict) and isinstance(v, dict):
            model_cfg[k].update(v)
        else:
            model_cfg[k] = v

    model_cfg['num_classes'] = num_classes

    if cache_dir is not None:
        os.environ.setdefault('HF_HOME', cache_dir)

    filtered_cfg = {k: v for k, v in model_cfg.items() if k in _CLASSIFICATION_VALID_PARAMS}
    model = ClassificationModel(**filtered_cfg)

    if pretrained:
        load_checkpoint(model, pretrained, strict=False)

    model = model.to(device)
    cast_dtype = get_cast_dtype(precision)
    if cast_dtype is not None:
        model = model.to(dtype=cast_dtype)

    return model
