"""Evaluation and attention-map extraction scripts."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import get_cifar10_loaders, get_raw_cifar10_test_dataset
from models.model import build_vit_from_config
from scripts.train import evaluate as evaluate_loop
from utils.visualization import (
    CIFAR10_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    save_attention_grid_panel,
    save_cls_attention_heatmap,
    save_full_attention_matrix,
)


def load_yaml(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(config: Dict | None = None) -> torch.device:
    requested = "auto"
    if config is not None:
        requested = str(config.get("training", {}).get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_scratch_model(checkpoint_path: str | Path, device: torch.device | None = None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    device = device or get_device(config)
    model = build_vit_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, checkpoint


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: str | Path) -> Dict:
    model, config, checkpoint = load_scratch_model(checkpoint_path)
    device = next(model.parameters()).device
    _, _, test_loader = get_cifar10_loaders(config)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("training", {}).get("label_smoothing", 0.0)))
    test_loss, test_acc = evaluate_loop(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.2f}%")
    return {"test_loss": test_loss, "test_acc": test_acc, "checkpoint": str(checkpoint_path)}


def _take_first_n_from_loader(loader, n: int, device: torch.device):
    images, labels = [], []
    for batch_images, batch_labels in loader:
        images.append(batch_images)
        labels.append(batch_labels)
        if sum(x.size(0) for x in images) >= n:
            break
    images = torch.cat(images, dim=0)[:n].to(device)
    labels = torch.cat(labels, dim=0)[:n].to(device)
    return images, labels


@torch.no_grad()
def save_scratch_attention_maps(
    checkpoint_path: str | Path,
    output_dir: str | Path = "outputs/attention_maps/scratch",
    layer_numbers: Sequence[int] = (1, 2, 4),
    head_index: int = 0,
    num_images: int = 2,
    use_logits: bool = False,
) -> List[Path]:
    """Save attention maps for the scratch ViT.

    layer_numbers are 1-indexed to match assignment wording.
    use_logits=False plots softmax attention probabilities; use_logits=True plots
    the raw scaled QK^T similarity matrix before softmax.
    """
    model, config, _ = load_scratch_model(checkpoint_path)
    device = next(model.parameters()).device
    _, _, test_loader = get_cifar10_loaders(config)
    images, labels = _take_first_n_from_loader(test_loader, num_images, device)

    logits, attention = model(images, return_attention=True)
    key = "logits" if use_logits else "probs"
    grid_size = model.grid_size
    output_dir = Path(output_dir)
    output_paths: List[Path] = []

    class_names = getattr(test_loader.dataset, "classes", None)

    for image_idx in range(num_images):
        panel_layer_maps = []

        for layer_number in layer_numbers:
            layer_idx = layer_number - 1
            if layer_idx < 0 or layer_idx >= len(attention[key]):
                raise ValueError(f"Requested layer {layer_number}; model has {len(attention[key])} layers")

            layer_attn = attention[key][layer_idx][image_idx].detach().cpu()  # (heads, N, N)
            single_head = layer_attn[head_index]
            avg_heads = layer_attn.mean(dim=0)

            panel_layer_maps.append(
                {
                    "layer": layer_number,
                    "single_head": single_head,
                    "mean_heads": avg_heads,
                }
            )

            # Keep the earlier separate files too, because they are useful for the report appendix.
            base = f"image{image_idx}_label{int(labels[image_idx])}_layer{layer_number}_{key}"
            output_paths.append(
                save_cls_attention_heatmap(
                    single_head,
                    grid_size=grid_size,
                    image=images[image_idx].detach().cpu(),
                    title=f"Layer {layer_number}, head {head_index}, CLS→patches",
                    output_path=output_dir / f"{base}_head{head_index}_cls.png",
                )
            )
            output_paths.append(
                save_cls_attention_heatmap(
                    avg_heads,
                    grid_size=grid_size,
                    image=images[image_idx].detach().cpu(),
                    title=f"Layer {layer_number}, average heads, CLS→patches",
                    output_path=output_dir / f"{base}_avg_heads_cls.png",
                )
            )
            output_paths.append(
                save_full_attention_matrix(
                    single_head,
                    title=f"Layer {layer_number}, head {head_index}, token-token matrix",
                    output_path=output_dir / f"{base}_head{head_index}_matrix.png",
                )
            )
            output_paths.append(
                save_full_attention_matrix(
                    avg_heads,
                    title=f"Layer {layer_number}, average heads, token-token matrix",
                    output_path=output_dir / f"{base}_avg_heads_matrix.png",
                )
            )

        label_int = int(labels[image_idx].detach().cpu())
        panel_key_name = "qk_logits" if use_logits else "attention_probs"
        output_paths.append(
            save_attention_grid_panel(
                image=images[image_idx].detach().cpu(),
                label=label_int,
                class_names=class_names if class_names is not None else (),
                layer_maps=panel_layer_maps,
                grid_size=grid_size,
                head_title="First Attention Head" if head_index == 0 else f"Attention Head {head_index}",
                mean_title="Mean Attention of Heads",
                output_path=output_dir / f"image{image_idx}_label{label_int}_{panel_key_name}_panel.png",
            )
        )

    for path in output_paths:
        print(f"Saved {path}")
    return output_paths


class TorchvisionViTAttentionExtractor:
    """Extract QK attention from torchvision ViT-B/16 using forward pre-hooks.

    Torchvision's encoder blocks call MultiheadAttention with need_weights=False,
    so a normal forward hook does not receive weights.  This helper captures the
    tensor entering each selected self_attention module and recomputes QK^T from
    that module's learned projection weights.
    """

    def __init__(self, model: nn.Module, layer_numbers: Sequence[int]) -> None:
        self.model = model
        self.layer_numbers = tuple(layer_numbers)
        self.attention: Dict[int, Dict[str, torch.Tensor]] = {}
        self.handles = []
        self._register_hooks()

    def _get_layer(self, layer_number: int):
        # torchvision names are encoder_layer_0 ... encoder_layer_11 for ViT-B/16.
        layer_idx = layer_number - 1
        layers = self.model.encoder.layers
        if hasattr(layers, f"encoder_layer_{layer_idx}"):
            return getattr(layers, f"encoder_layer_{layer_idx}")
        return layers[layer_idx]

    def _register_hooks(self) -> None:
        for layer_number in self.layer_numbers:
            block = self._get_layer(layer_number)
            handle = block.self_attention.register_forward_pre_hook(
                self._make_hook(layer_number), with_kwargs=False
            )
            self.handles.append(handle)

    def _make_hook(self, layer_number: int):
        def hook(module: nn.MultiheadAttention, inputs):
            # inputs: query, key, value; ViT self-attention uses the same tensor.
            x = inputs[0]
            if not module.batch_first:
                x = x.transpose(0, 1)
            embed_dim = module.embed_dim
            num_heads = module.num_heads
            head_dim = embed_dim // num_heads

            q_proj_weight = module.in_proj_weight[:embed_dim]
            k_proj_weight = module.in_proj_weight[embed_dim : 2 * embed_dim]
            q_proj_bias = module.in_proj_bias[:embed_dim] if module.in_proj_bias is not None else None
            k_proj_bias = module.in_proj_bias[embed_dim : 2 * embed_dim] if module.in_proj_bias is not None else None

            q = F.linear(x, q_proj_weight, q_proj_bias)
            k = F.linear(x, k_proj_weight, k_proj_bias)
            batch_size, seq_len, _ = q.shape
            q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            probs = logits.softmax(dim=-1)
            self.attention[layer_number] = {"logits": logits.detach().cpu(), "probs": probs.detach().cpu()}

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _pil_to_tensor_batch(images: Sequence[Image.Image], transform, device: torch.device) -> torch.Tensor:
    tensors = [transform(img) for img in images]
    return torch.stack(tensors, dim=0).to(device)


@torch.no_grad()
def save_pretrained_vit_attention_maps(
    config_path: str | Path = "config/config.yaml",
    output_dir: str | Path = "outputs/attention_maps/pretrained_vit_b_16",
    layer_numbers: Sequence[int] = (4, 8, 12),
    head_index: int = 0,
    image_indices: Sequence[int] = (0, 1),
    use_logits: bool = False,
) -> List[Path]:
    """Save attention maps for torchvision's pretrained ViT-B/16 on CIFAR-10 images.

    This also saves one report-friendly panel per image:
        rows    = selected layers, e.g. L4, L8, L12
        columns = patched input, selected attention head, mean attention of heads
    """
    config = load_yaml(config_path)
    device = get_device(config)

    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights).to(device).eval()
    transform = weights.transforms()

    raw_test_dataset = get_raw_cifar10_test_dataset(config)
    pil_images = [raw_test_dataset[i][0] for i in image_indices]
    labels = [raw_test_dataset[i][1] for i in image_indices]
    class_names = getattr(raw_test_dataset, "classes", CIFAR10_CLASSES)

    batch = _pil_to_tensor_batch(pil_images, transform, device)

    extractor = TorchvisionViTAttentionExtractor(model, layer_numbers=layer_numbers)
    _ = model(batch)
    extractor.close()

    output_dir = Path(output_dir)
    output_paths: List[Path] = []
    key = "logits" if use_logits else "probs"

    # Do not hard-code 14x14. Compute it from the token count.
    # N = 1 CLS token + patch tokens.
    first_layer = layer_numbers[0]
    num_tokens = extractor.attention[first_layer][key].shape[-1]
    num_patches = num_tokens - 1
    grid_size = int(math.sqrt(num_patches))

    if grid_size * grid_size != num_patches:
        raise ValueError(
            f"Cannot reshape {num_patches} patch tokens into a square grid."
        )

    for batch_idx, original_index in enumerate(image_indices):
        panel_layer_maps = []

        for layer_number in layer_numbers:
            if layer_number not in extractor.attention:
                raise ValueError(f"Layer {layer_number} was not captured by the attention extractor.")

            layer_attn = extractor.attention[layer_number][key][batch_idx]  # (heads, N, N)

            if head_index < 0 or head_index >= layer_attn.size(0):
                raise ValueError(
                    f"head_index={head_index} is invalid; this layer has {layer_attn.size(0)} heads."
                )

            single_head = layer_attn[head_index]
            avg_heads = layer_attn.mean(dim=0)

            panel_layer_maps.append(
                {
                    "layer": layer_number,
                    "single_head": single_head,
                    "mean_heads": avg_heads,
                }
            )

            # Keep the detailed individual outputs too.
            base = f"cifar_index{original_index}_label{labels[batch_idx]}_layer{layer_number}_{key}"

            output_paths.append(
                save_cls_attention_heatmap(
                    single_head,
                    grid_size=grid_size,
                    title=f"Pretrained ViT-B/16 layer {layer_number}, head {head_index}",
                    output_path=output_dir / f"{base}_head{head_index}_cls.png",
                )
            )

            output_paths.append(
                save_cls_attention_heatmap(
                    avg_heads,
                    grid_size=grid_size,
                    title=f"Pretrained ViT-B/16 layer {layer_number}, average heads",
                    output_path=output_dir / f"{base}_avg_heads_cls.png",
                )
            )

            output_paths.append(
                save_full_attention_matrix(
                    single_head,
                    title=f"Pretrained ViT-B/16 layer {layer_number}, head {head_index}",
                    output_path=output_dir / f"{base}_head{head_index}_matrix.png",
                )
            )

            output_paths.append(
                save_full_attention_matrix(
                    avg_heads,
                    title=f"Pretrained ViT-B/16 layer {layer_number}, average heads",
                    output_path=output_dir / f"{base}_avg_heads_matrix.png",
                )
            )

        label_int = int(labels[batch_idx])
        panel_key_name = "qk_logits" if use_logits else "attention_probs"

        output_paths.append(
            save_attention_grid_panel(
                image=batch[batch_idx].detach().cpu(),
                label=label_int,
                class_names=class_names,
                layer_maps=panel_layer_maps,
                grid_size=grid_size,
                head_title="First Attention Head" if head_index == 0 else f"Attention Head {head_index}",
                mean_title="Mean Attention of Heads",
                output_path=output_dir
                / f"cifar_index{original_index}_label{label_int}_{panel_key_name}_panel.png",
                image_mean=IMAGENET_MEAN,
                image_std=IMAGENET_STD,
            )
        )

    for path in output_paths:
        print(f"Saved {path}")

    return output_paths
