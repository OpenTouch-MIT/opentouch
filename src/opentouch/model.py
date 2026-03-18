"""CrossRetrievalModel: cross-modal contrastive model for visual-tactile-pose retrieval."""

import logging
import math
from typing import Dict, List, Optional, Set

import torch
import torch.nn.functional as F
from torch import nn

from .visual_encoder import DINOv3Encoder, ResNetEncoder
from .tactile_encoder import CNNetEmbedding
from .pose_encoder import PoseEncoder

logger = logging.getLogger(__name__)


def get_cast_dtype(precision: str):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }.get(precision)


def get_input_dtype(precision: str):
    return {
        "fp16": torch.float16,
        "pure_fp16": torch.float16,
        "bf16": torch.bfloat16,
        "pure_bf16": torch.bfloat16,
    }.get(precision)


class CrossRetrievalModel(nn.Module):
    """Cross-modal retrieval model """

    # target_modality -> (fusion_attr, (source_mod_a, source_mod_b))
    _FUSION_MAP = {
        "pose": ("tactile_visual_fusion", ("tactile", "visual")),
        "tactile": ("pose_visual_fusion", ("pose", "visual")),
        "visual": ("tactile_pose_fusion", ("tactile", "pose")),
    }

    def __init__(
        self,
        embed_dim: int = 64,
        visual_cfg: Optional[dict] = None,
        tactile_cfg: Optional[dict] = None,
        pose_cfg: Optional[dict] = None,
        enabled_modalities: Optional[List[str]] = None,
        fusion_method: str = "concat",
        normalize: bool = True,
        init_logit_scale: Optional[float] = None,
        init_logit_bias: Optional[float] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_method = fusion_method
        self.normalize = normalize
        self.enabled_modalities: Set[str] = (
            set(enabled_modalities) if enabled_modalities is not None
            else {"visual", "tactile", "pose"}
        )

        visual_cfg = dict(visual_cfg) if visual_cfg else {}
        tactile_cfg = dict(tactile_cfg) if tactile_cfg else {}
        pose_cfg = dict(pose_cfg) if pose_cfg else {}
        self._init_encoders(visual_cfg, tactile_cfg, pose_cfg)
        self._init_fusion_modules()

        if init_logit_scale is None:
            init_logit_scale = math.log(1 / 0.07)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.logit_bias = (nn.Parameter(torch.ones([]) * init_logit_bias) if init_logit_bias is not None else None)


    def _init_encoders(self, visual_cfg: dict, tactile_cfg: dict, pose_cfg: dict) -> None:
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


    def _init_fusion_modules(self) -> None:
        """Create fusion projections for multi-modal queries (3+ modalities, concat only)."""
        self.tactile_visual_fusion: Optional[nn.Module] = None
        self.pose_visual_fusion: Optional[nn.Module] = None
        self.tactile_pose_fusion: Optional[nn.Module] = None
        if len(self.enabled_modalities) < 3:
            return
        if self.fusion_method != "concat":
            logger.warning(f"Fusion '{self.fusion_method}' not supported, only 'concat' implemented.")
            return
        if {"tactile", "visual"}.issubset(self.enabled_modalities):
            self.tactile_visual_fusion = nn.Linear(self.embed_dim * 2, self.embed_dim)
        if {"pose", "visual"}.issubset(self.enabled_modalities):
            self.pose_visual_fusion = nn.Linear(self.embed_dim * 2, self.embed_dim)
        if {"tactile", "pose"}.issubset(self.enabled_modalities):
            self.tactile_pose_fusion = nn.Linear(self.embed_dim * 2, self.embed_dim)


    def _normalize_embedding(self, emb: torch.Tensor) -> torch.Tensor:
        return F.normalize(emb, dim=-1) if self.normalize else emb


    def _encode_modality(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Encode a single modality and optionally L2-normalize."""
        encoder = getattr(self, name)
        if encoder is None:
            raise RuntimeError("Invalid modality input.")
        return self._normalize_embedding(encoder(x))


    def encode_visual(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T,C,H,W] or [B,C,H,W] -> [B, embed_dim]"""
        return self._encode_modality("visual", x)


    def encode_tactile(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T,1,16,16] or [B,T,16,16] -> [B, embed_dim]"""
        return self._encode_modality("tactile", x)


    def encode_pose(self, landmarks: torch.Tensor) -> torch.Tensor:
        """[B,T,21,3] -> [B, embed_dim]"""
        return self._encode_modality("pose", landmarks)


    def _resolve_fusion(self, target_modality: str):
        """Look up fusion module and source modalities for a target."""
        if target_modality not in self._FUSION_MAP:
            raise ValueError("Invalid modality input.")
        attr, sources = self._FUSION_MAP[target_modality]
        module = getattr(self, attr)
        if module is None:
            raise RuntimeError("Invalid modality input.")
        return module, sources

    def _append_encoded_if_available(
        self,
        out: Dict[str, torch.Tensor],
        *,
        input_tensor: Optional[torch.Tensor],
        encoder: Optional[nn.Module],
        feature_key: str,
        encode_fn,
    ) -> None:
        if input_tensor is not None and encoder is not None:
            out[feature_key] = encode_fn(input_tensor)

    def encode_multimodal_query(
        self,
        modalities_dict: Dict[str, torch.Tensor],
        target_modality: str,
    ) -> torch.Tensor:
        """Fuse two raw-input modalities to query the third."""
        module, (mod_a, mod_b) = self._resolve_fusion(target_modality)
        emb_a = self._encode_modality(mod_a, modalities_dict[mod_a])
        emb_b = self._encode_modality(mod_b, modalities_dict[mod_b])
        fused = module(torch.cat([emb_a, emb_b], dim=-1))
        return self._normalize_embedding(fused)

    def fuse_encoded_features(
        self,
        encoded_features: Dict[str, torch.Tensor],
        target_modality: str,
    ) -> torch.Tensor:
        """Fuse pre-computed embeddings for multi-modal querying."""
        module, (mod_a, mod_b) = self._resolve_fusion(target_modality)
        fused = module(torch.cat([encoded_features[mod_a], encoded_features[mod_b]], dim=-1))
        return self._normalize_embedding(fused)


    def forward(
        self,
        rgb_images: Optional[torch.Tensor] = None,
        tactile_pressure: Optional[torch.Tensor] = None,
        hand_landmarks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode all provided modalities, return dict of features + logit_scale."""
        out: Dict[str, torch.Tensor] = {}
        self._append_encoded_if_available(
            out, input_tensor=rgb_images, encoder=self.visual,
            feature_key="visual_features", encode_fn=self.encode_visual,
        )
        self._append_encoded_if_available(
            out, input_tensor=tactile_pressure, encoder=self.tactile,
            feature_key="tactile_features", encode_fn=self.encode_tactile,
        )
        self._append_encoded_if_available(
            out, input_tensor=hand_landmarks, encoder=self.pose,
            feature_key="pose_features", encode_fn=self.encode_pose,
        )
        out["logit_scale"] = self.logit_scale.exp()
        if self.logit_bias is not None:
            out["logit_bias"] = self.logit_bias
        return out
