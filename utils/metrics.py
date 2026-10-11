"""Metrics and bookkeeping helpers."""
from __future__ import annotations

from typing import Iterable

import torch


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy in percent."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item() * 100.0


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    params: Iterable[torch.nn.Parameter] = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def token_count(image_size: int, patch_size: int) -> int:
    """Number of ViT tokens including CLS."""
    return (image_size // patch_size) ** 2 + 1


def attention_matrix_size(image_size: int, patch_size: int, depth: int, heads: int) -> int:
    """Proxy for attention cost: total elements across all attention matrices."""
    n = token_count(image_size, patch_size)
    return depth * heads * n * n


def format_large_number(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)
