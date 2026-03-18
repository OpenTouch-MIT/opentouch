"""Retrieval metrics computation utilities."""

from typing import Dict, List, Optional
import torch
import torch.nn.functional as F


def _sanitize_label(label: str) -> str:
    """Normalize modality label for dictionary keys."""
    return label.strip().lower().replace(" ", "_").replace("+", "_")


def compute_retrieval_metrics(
    query_emb: torch.Tensor,
    target_emb: torch.Tensor,
    top_k: Optional[List[int]] = None,
    *,
    query_label: str = "query",
    target_label: str = "target",
    compute_reverse: bool = True,
    use_map: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Compute retrieval metrics for the specified query/target modalities."""
    top_k = top_k or [1, 5, 10]
    query_emb, target_emb = _promote_to_common_dtype(query_emb, target_emb)
    query_emb = F.normalize(query_emb, dim=1)
    target_emb = F.normalize(target_emb, dim=1)

    num_samples = len(query_emb)
    if num_samples == 0:
        raise ValueError("Invalid retrieval inputs.")

    max_k = min(max(top_k), num_samples)
    similarity_q2t = query_emb @ target_emb.t()
    top_indices_q2t = _topk_indices(similarity_q2t, max_k)

    query_key, target_key = _sanitize_label(query_label), _sanitize_label(target_label)
    metrics: Dict[str, Dict[str, float]] = {
        f"{query_key}_to_{target_key}": _compute_direction_metrics(
            top_indices_q2t, top_k, num_samples, use_map, similarity=similarity_q2t,
        )
    }

    if compute_reverse:
        similarity_t2q = target_emb @ query_emb.t()
        top_indices_t2q = _topk_indices(similarity_t2q, max_k)
        metrics[f"{target_key}_to_{query_key}"] = _compute_direction_metrics(
            top_indices_t2q, top_k, num_samples, use_map, similarity=similarity_t2q,
        )

    return metrics


def _compute_direction_metrics(
    top_indices: torch.Tensor,
    top_k: List[int],
    num_samples: int,
    use_map: bool = True,
    similarity: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute recall@k and mAP."""
    results: Dict[str, float] = {}
    target_indices = torch.arange(num_samples, device=top_indices.device).unsqueeze(1)

    for k in top_k:
        if k > num_samples:
            results[f"recall@{k}"] = 1.0
            continue
        correct = (top_indices[:, :k] == target_indices).sum().item()
        results[f"recall@{k}"] = correct / num_samples

    if use_map:
        if similarity is None:
            raise ValueError("Invalid retrieval inputs.")
        correct_sims = similarity.diag().unsqueeze(1)
        ranks = (similarity >= correct_sims).sum(dim=1)
        results["mAP"] = (1.0 / ranks.float()).mean().item()

    return results


def _promote_to_common_dtype(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    common_dtype = torch.promote_types(a.dtype, b.dtype)
    if a.dtype != common_dtype:
        a = a.to(common_dtype)
    if b.dtype != common_dtype:
        b = b.to(common_dtype)
    return a, b


def _topk_indices(similarity: torch.Tensor, k: int) -> torch.Tensor:
    _, indices = torch.topk(similarity, k, dim=1)
    return indices
