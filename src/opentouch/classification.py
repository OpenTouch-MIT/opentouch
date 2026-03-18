"""ClassificationModel: multi-modal classification model with shared encoders."""

import logging
from typing import List, Optional, Set

import torch
import torch.nn.functional as F
from torch import nn

from .visual_encoder import DINOv3Encoder, ResNetEncoder
from .tactile_encoder import CNNetEmbedding
from .pose_encoder import PoseEncoder

logger = logging.getLogger(__name__)


class ClassificationModel(nn.Module):
    """Multi-modal classification model that reuses visual, tactile, and pose
    encoders with fusion and a classification head."""

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 64,
        visual_cfg: Optional[dict] = None,
        tactile_cfg: Optional[dict] = None,
        pose_cfg: Optional[dict] = None,
        enabled_modalities: Optional[List[str]] = None,
        fusion_method: str = "concat",
        normalize: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_method = fusion_method
        self.normalize = normalize
        self.enabled_modalities: Set[str] = (
            set(enabled_modalities)
            if enabled_modalities is not None
            else {"visual", "tactile", "pose"}
        )

        self.num_classes = num_classes

        visual_cfg = dict(visual_cfg) if visual_cfg else {}
        tactile_cfg = dict(tactile_cfg) if tactile_cfg else {}
        pose_cfg = dict(pose_cfg) if pose_cfg else {}

        self._init_encoders(visual_cfg, tactile_cfg, pose_cfg)
        self._init_fusion()
        self._init_classification_head()

    def _init_encoders(self, visual_cfg: dict, tactile_cfg: dict, pose_cfg: dict) -> None:
        """Initialize encoders following the same pattern as CrossRetrievalModel."""
        if "visual" in self.enabled_modalities:
            encoder_type = visual_cfg.pop("encoder_type", "dinov3")
            if encoder_type.startswith("resnet"):
                backbone = visual_cfg.pop("backbone", encoder_type)
                self.visual = ResNetEncoder(embed_dim=self.embed_dim, backbone=backbone, **visual_cfg)
            else:
                self.visual = DINOv3Encoder(embed_dim=self.embed_dim, **visual_cfg)
        else:
            self.visual = None

        if "tactile" in self.enabled_modalities:
            encoder_type = tactile_cfg.pop("encoder_type", "cnnet")
            if encoder_type == "cnnet":
                self.tactile = CNNetEmbedding(emb_dim=self.embed_dim, **tactile_cfg)
            else:
                raise ValueError("Invalid model configuration.")
        else:
            self.tactile = None

        if "pose" in self.enabled_modalities:
            pose_cfg.pop("encoder_type", None)
            self.pose = PoseEncoder(emb_dim=self.embed_dim, **pose_cfg)
        else:
            self.pose = None

    def _init_fusion(self) -> None:
        """Create fusion module based on the number of enabled modalities.

        Single modality: Identity (no fusion needed).
        Multiple modalities: concat embeddings then project back to embed_dim.
        """
        n_modalities = len(self.enabled_modalities)
        if n_modalities <= 1:
            self.fusion = nn.Identity()
            self._feature_dim = self.embed_dim
        elif self.fusion_method == "concat":
            concat_dim = self.embed_dim * n_modalities
            self.fusion = nn.Linear(concat_dim, self.embed_dim)
            self._feature_dim = self.embed_dim
        else:
            raise ValueError("Invalid model configuration.")

    def _init_classification_head(self) -> None:
        """Create classification head."""
        self.head = nn.Linear(self._feature_dim, self.num_classes)

    def _normalize_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            return F.normalize(embedding, dim=-1)
        return embedding

    def _encode_and_append(
        self,
        embeddings: List[torch.Tensor],
        module: Optional[nn.Module],
        tensor: Optional[torch.Tensor],
    ) -> None:
        """Encode and append one modality when both module and tensor are present."""
        if module is None or tensor is None:
            return
        embeddings.append(self._normalize_embedding(module(tensor)))

    def encode_modalities(
        self,
        rgb_images: Optional[torch.Tensor] = None,
        tactile_pressure: Optional[torch.Tensor] = None,
        hand_landmarks: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Encode each enabled modality, optionally L2-normalize, return list of embeddings.

        The returned list preserves a consistent ordering: visual, tactile, pose
        (only those that are both enabled and provided).
        """
        embeddings: List[torch.Tensor] = []
        self._encode_and_append(embeddings, self.visual, rgb_images)
        self._encode_and_append(embeddings, self.tactile, tactile_pressure)
        self._encode_and_append(embeddings, self.pose, hand_landmarks)

        return embeddings

    def forward(
        self,
        rgb_images: Optional[torch.Tensor] = None,
        tactile_pressure: Optional[torch.Tensor] = None,
        hand_landmarks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode inputs, fuse modalities, and apply classification head.

        Returns logits tensor of shape (B, num_classes).
        """
        embeddings = self.encode_modalities(rgb_images, tactile_pressure, hand_landmarks)

        if len(embeddings) == 0:
            raise RuntimeError("Invalid modality input.")

        if len(embeddings) != len(self.enabled_modalities):
            raise RuntimeError("Invalid modality input.")

        fused_input = embeddings[0] if len(embeddings) == 1 else torch.cat(embeddings, dim=-1)
        fused = self.fusion(fused_input)

        return self.head(fused)
