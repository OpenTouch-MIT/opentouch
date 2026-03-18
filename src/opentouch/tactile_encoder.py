from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_TACTILE_HW = (16, 16)

# Normalize helper
def _normalize_input(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """[B, T, 16, 16] or [B, T, 1, 16, 16] -> [B*T, 1, 16, 16], returns (flat, B, T)."""
    if x.dim() == 5:
        b, t, c, h, w = x.shape
        is_valid = c == 1
        if is_valid:
            x = x.squeeze(2)
    elif x.dim() == 4:
        b, t, h, w = x.shape
        is_valid = True
    else:
        is_valid = False

    if not is_valid or (h, w) != _TACTILE_HW:
        raise ValueError("Invalid modality input.")
    return x.reshape(b * t, 1, h, w), b, t


class CNNetEmbedding(nn.Module):
    def __init__(self, emb_dim: int = 32) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.gru = nn.GRU(128, 120, num_layers=2, bidirectional=True)
        self.projection = nn.Linear(240, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat, b, t = _normalize_input(x)
        feats = self.cnn(flat)
        seq = feats.view(b, t, -1).transpose(0, 1)
        self.gru.flatten_parameters()
        out, _ = self.gru(seq)
        combined = torch.cat([out[-1, :, :120], out[0, :, 120:]], dim=1)
        return self.projection(F.relu(combined))
