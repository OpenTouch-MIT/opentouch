"""Classification metrics computation utilities."""

from typing import Dict
import torch


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    prefix: str = "",
) -> Dict[str, float]:
    """Compute accuracy, macro F1, and mean per-class accuracy."""
    preds = logits.argmax(dim=1)
    n = labels.shape[0]

    accuracy = (preds == labels).sum().item() / n if n > 0 else 0.0

    per_class_acc = []
    precisions = []
    recalls = []

    for c in range(num_classes):
        mask_true = labels == c
        mask_pred = preds == c

        tp = (mask_true & mask_pred).sum().item()
        fp = (~mask_true & mask_pred).sum().item()
        fn = (mask_true & ~mask_pred).sum().item()
        class_count = mask_true.sum().item()
        if class_count > 0:
            per_class_acc.append(tp / class_count)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)

    f1_scores = []
    for p, r in zip(precisions, recalls):
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1_scores.append(f1)
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    mean_per_class_acc = sum(per_class_acc) / len(per_class_acc) if per_class_acc else 0.0

    return {
        f"{prefix}accuracy": accuracy,
        f"{prefix}macro_f1": macro_f1,
        f"{prefix}mean_per_class_accuracy": mean_per_class_acc,
    }
