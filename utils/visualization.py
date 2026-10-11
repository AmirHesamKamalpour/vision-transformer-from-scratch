"""Plotting utilities for training curves and attention maps."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch

# CIFAR-10 normalization constants used for denormalization in plots.
CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def denormalize_cifar10(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().cpu() * CIFAR10_STD + CIFAR10_MEAN
    return image.clamp(0, 1)

def denormalize_with_stats(
    image: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """Denormalize a CHW image tensor using custom mean/std."""
    image = image.detach().cpu()

    if image.ndim != 3 or image.size(0) != 3:
        raise ValueError("Expected image shape (3, H, W)")

    mean_t = torch.tensor(mean, dtype=image.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype).view(3, 1, 1)

    return (image * std_t + mean_t).clamp(0, 1)


def plot_training_curves(history_csv: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    """Save loss and accuracy plots from a CSV history file."""
    history_csv = Path(history_csv)
    output_dir = ensure_dir(output_dir)
    df = pd.read_csv(history_csv)

    loss_path = output_dir / f"{history_csv.stem}_loss.png"
    acc_path = output_dir / f"{history_csv.stem}_accuracy.png"

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="train")
    plt.plot(df["epoch"], df["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training/validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_acc"], label="train")
    plt.plot(df["epoch"], df["val_acc"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training/validation accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path, dpi=200)
    plt.close()

    return loss_path, acc_path


def _cls_attention_to_grid(attention: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Convert CLS-token row of a token-token matrix to a patch-grid heatmap."""
    if attention.ndim == 3:
        attention = attention.mean(dim=0)
    if attention.ndim != 2:
        raise ValueError("attention must have shape (N, N) or (heads, N, N)")

    cls_to_patches = attention[0, 1:]  # CLS query row, remove CLS-to-CLS entry.
    return cls_to_patches.reshape(grid_size, grid_size).detach().cpu()


def _minmax_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.detach().cpu().float()
    return (x - x.min()) / (x.max() - x.min() + eps)


def _hide_ticks_and_spines(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_patch_grid(ax, image: torch.Tensor, grid_size: int) -> None:
    """Draw thin patch boundaries on top of an image axis."""
    height, width = image.shape[:2]
    step_y = height / grid_size
    step_x = width / grid_size
    # Draw interior boundaries only. Use a subtle dark line like the reference image.
    for i in range(1, grid_size):
        ax.axhline(i * step_y - 0.5, linewidth=0.45, alpha=0.45, color="black")
        ax.axvline(i * step_x - 0.5, linewidth=0.45, alpha=0.45, color="black")


def save_cls_attention_heatmap(
    attention: torch.Tensor,
    grid_size: int,
    output_path: str | Path,
    title: str,
    image: Optional[torch.Tensor] = None,
) -> Path:
    """Save CLS-token attention over image patches.

    Args:
        attention: Either (N, N) for one head or (heads, N, N) for multiple heads.
        grid_size: Number of patches along one image dimension.
        image: Optional normalized CIFAR image, shape (C, H, W), plotted beside heatmap.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    heatmap = _cls_attention_to_grid(attention, grid_size)

    if image is None:
        plt.figure(figsize=(5, 4))
        plt.imshow(heatmap)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(title)
        plt.axis("off")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        img = denormalize_cifar10(image).permute(1, 2, 0).numpy()
        axes[0].imshow(img)
        axes[0].set_title("Input image")
        axes[0].axis("off")
        im = axes[1].imshow(heatmap)
        axes[1].set_title(title)
        axes[1].axis("off")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_attention_grid_panel(
    image: torch.Tensor,
    label: int,
    layer_maps: Sequence[dict],
    grid_size: int,
    output_path: str | Path,
    class_names: Sequence[str] = CIFAR10_CLASSES,
    head_title: str = "First Attention Head",
    mean_title: str = "Mean Attention of Heads",
    cmap: str = "YlGn",
    normalize_each_map: bool = True,
    image_mean: Optional[Sequence[float]] = None,
    image_std: Optional[Sequence[float]] = None,
) -> Path:
    """Save one Part-2b figure in the same layout as the requested example.

    The figure has one row per requested layer and three columns:
    patched input image, selected-head CLS attention, and mean-head CLS attention.

    Each item in layer_maps must contain:
        {
            "layer": 1,
            "single_head": tensor(N, N),
            "mean_heads": tensor(N, N),
        }
    """
    if not layer_maps:
        raise ValueError("layer_maps cannot be empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class_name = class_names[label] if 0 <= int(label) < len(class_names) else str(label)
    
    if image_mean is None or image_std is None:
        plot_tensor = denormalize_cifar10(image)
    else:
        plot_tensor = denormalize_with_stats(image, image_mean, image_std)
    img = plot_tensor.permute(1, 2, 0).numpy()

    n_rows = len(layer_maps)

    fig, axes = plt.subplots(n_rows, 3, figsize=(9.0, 3.0 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 3)

    for row_idx, layer_info in enumerate(layer_maps):
        layer_number = int(layer_info["layer"])
        single_grid = _cls_attention_to_grid(layer_info["single_head"], grid_size)
        mean_grid = _cls_attention_to_grid(layer_info["mean_heads"], grid_size)
        if normalize_each_map:
            single_grid = _minmax_normalize(single_grid)
            mean_grid = _minmax_normalize(mean_grid)

        # Column 1: image with patch grid.
        ax_img = axes[row_idx, 0]
        ax_img.imshow(img)
        _draw_patch_grid(ax_img, img, grid_size)
        ax_img.set_title(f"Patched Input {class_name}", fontsize=10)
        _hide_ticks_and_spines(ax_img)
        ax_img.set_ylabel(
            f"L{layer_number}",
            rotation=0,
            labelpad=22,
            fontsize=11,
            fontweight="bold",
            va="center",
        )

        # Column 2: one arbitrary / selected head.
        ax_head = axes[row_idx, 1]
        ax_head.imshow(single_grid, cmap=cmap, interpolation="nearest")
        ax_head.set_title(head_title, fontsize=10)
        _hide_ticks_and_spines(ax_head)

        # Column 3: mean over all heads.
        ax_mean = axes[row_idx, 2]
        ax_mean.imshow(mean_grid, cmap=cmap, interpolation="nearest")
        ax_mean.set_title(mean_title, fontsize=10)
        _hide_ticks_and_spines(ax_mean)

    plt.tight_layout(w_pad=2.2, h_pad=1.4)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_full_attention_matrix(
    attention: torch.Tensor,
    output_path: str | Path,
    title: str,
) -> Path:
    """Save the full token-token attention/similarity matrix."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if attention.ndim == 3:
        attention = attention.mean(dim=0)
    if attention.ndim != 2:
        raise ValueError("attention must have shape (N, N) or (heads, N, N)")

    plt.figure(figsize=(5, 4))
    plt.imshow(attention.detach().cpu())
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlabel("Key token index")
    plt.ylabel("Query token index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path

