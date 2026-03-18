from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel


def freeze(module: nn.Module, set_eval_mode: bool = False) -> None:
    """Freeze module parameters and optionally set to eval mode."""
    for p in module.parameters():
        p.requires_grad = False
    if set_eval_mode:
        module.eval()


def unfreeze(module: nn.Module) -> None:
    """Unfreeze module parameters and set back to training mode."""
    for p in module.parameters():
        p.requires_grad = True
    module.train()


def flatten_video(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """[B, T, C, H, W] -> [B*T, C, H, W], returns (flat, B, T)."""
    b, t, c, h, w = x.shape
    return x.reshape(b * t, c, h, w), b, t


def to_video_batch(x: torch.Tensor) -> torch.Tensor:
    """Normalize input to [B, T, C, H, W]."""
    if x.dim() == 4:
        return x.unsqueeze(1)
    if x.dim() == 5:
        return x
    raise ValueError("Invalid modality input.")


class TemporalPool(nn.Module):
    METHODS = {"mean", "first", "attn"}

    def __init__(self, method: str, dim: int) -> None:
        super().__init__()
        if method not in self.METHODS:
            raise ValueError("Invalid model configuration.")
        self.method = method
        self._scale = dim ** 0.5
        if method == "attn":
            self.query = nn.Parameter(torch.zeros(1, dim))
            nn.init.normal_(self.query, std=0.02)
            self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "mean":
            return x.mean(dim=1)
        if self.method == "first":
            return x[:, 0, :]
        q = self.proj(self.query).expand(x.size(0), -1)
        attn_scores = (x @ q.unsqueeze(-1)).squeeze(-1) / self._scale
        attn = torch.softmax(attn_scores, dim=1)
        return (attn.unsqueeze(-1) * x).sum(dim=1)


class ResNetEncoder(nn.Module):
    """ResNet-based visual encoder with temporal pooling."""
    _WEIGHTS = {
        "resnet18": (models.resnet18, "IMAGENET1K_V1", 512),
        "resnet50": (models.resnet50, "IMAGENET1K_V1", 2048),
    }

    def __init__(self, embed_dim: int, backbone: str = "resnet50",
                 freeze_backbone: bool = True, time_pool: str = "mean") -> None:
        super().__init__()
        builder, weight_enum, feat_dim = self._WEIGHTS[backbone]
        self.backbone = nn.Sequential(*list(builder(weights=weight_enum).children())[:-1])
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            freeze(self.backbone, set_eval_mode=True)

        self.temporal_pool = TemporalPool(time_pool, feat_dim)
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2, bias=False),
            nn.BatchNorm1d(feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, embed_dim, bias=True),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = to_video_batch(x)
        flat, b, t = flatten_video(x)
        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.backbone(flat).flatten(1)
        else:
            feats = self.backbone(flat).flatten(1)
        per_frame = feats.view(b, t, -1)
        return self.projection(self.temporal_pool(per_frame).float())


class DINOv3Encoder(nn.Module):
    """DINOv3-based visual encoder with temporal pooling."""
    def __init__(
        self,
        *,
        embed_dim: int,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        freeze_backbone: bool = True,
        time_pool: str = "mean",
    ) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            freeze(self.model, set_eval_mode=True)
        hidden_size = self.model.config.hidden_size
        self.temporal_pool = TemporalPool(time_pool, hidden_size)
        self.projection = nn.Linear(hidden_size, embed_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.model.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = to_video_batch(x)
        flat, b, t = flatten_video(x)
        if self.freeze_backbone:
            with torch.no_grad():
                cls_tokens = self.model(pixel_values=flat).last_hidden_state[:, 0, :]
        else:
            cls_tokens = self.model(pixel_values=flat).last_hidden_state[:, 0, :]
        per_frame = cls_tokens.view(b, t, -1)
        return self.projection(self.temporal_pool(per_frame).float())
