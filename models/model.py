"""Vision Transformer implementation from scratch for CIFAR-10.

Allowed dependencies here are basic PyTorch tensor/layer primitives.  The ViT
used for Parts 1-2 does NOT use torchvision.models, nn.TransformerEncoder, or
nn.MultiheadAttention.  Q, K, V projection and scaled dot-product attention are
implemented explicitly so attention maps can be inspected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


@dataclass
class ViTConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 8
    mlp_dim: int = 128
    dropout: float = 0.1
    attention_dropout: float = 0.1
    qkv_bias: bool = True


def _to_config(config: Dict | ViTConfig) -> ViTConfig:
    if isinstance(config, ViTConfig):
        return config
    return ViTConfig(**config)


class PatchEmbedding(nn.Module):
    """Split image into non-overlapping patches and project patches to tokens.

    A Conv2d with kernel_size=stride=patch_size is equivalent to flattening each
    patch and applying a shared linear projection, but is simpler and faster.
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, embed_dim, grid, grid) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    """Explicit multi-head self-attention.

    Returns attention logits and probabilities when requested. Shapes:
        x:             (B, N, C)
        attn_logits:   (B, heads, N, N)   before softmax
        attn_probs:    (B, heads, N, N)   after softmax/dropout
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = attn_logits.softmax(dim=-1)
        attn_probs_dropped = self.attn_drop(attn_probs)

        out = torch.matmul(attn_probs_dropped, v)  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        out = self.proj(out)
        out = self.proj_drop(out)

        if return_attention:
            return out, attn_logits, attn_probs
        return out


class TransformerEncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block for ViT."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
            qkv_bias=qkv_bias,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim=embed_dim, mlp_dim=mlp_dim, dropout=dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if return_attention:
            attn_out, attn_logits, attn_probs = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn_logits, attn_probs

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Small ViT classifier for CIFAR-10."""

    def __init__(self, config: Dict | ViTConfig) -> None:
        super().__init__()
        self.config = _to_config(config)
        cfg = self.config

        self.patch_embed = PatchEmbedding(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dim,
        )
        num_tokens = self.patch_embed.num_patches + 1  # +1 for CLS token

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, cfg.embed_dim))
        self.pos_drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    mlp_dim=cfg.mlp_dim,
                    dropout=cfg.dropout,
                    attention_dropout=cfg.attention_dropout,
                    qkv_bias=cfg.qkv_bias,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

        self._init_weights()

    @property
    def num_patches(self) -> int:
        return self.patch_embed.num_patches

    @property
    def grid_size(self) -> int:
        return self.patch_embed.grid_size

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if not return_attention:
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            return x[:, 0]

        attention: Dict[str, List[torch.Tensor]] = {"logits": [], "probs": []}
        for block in self.blocks:
            x, attn_logits, attn_probs = block(x, return_attention=True)
            attention["logits"].append(attn_logits)
            attention["probs"].append(attn_probs)
        x = self.norm(x)
        return x[:, 0], attention

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        if return_attention:
            features, attention = self.forward_features(x, return_attention=True)
            logits = self.head(features)
            return logits, attention
        features = self.forward_features(x)
        return self.head(features)


def build_vit_from_config(config: Dict) -> VisionTransformer:
    """Factory used by training/evaluation scripts."""
    model_cfg = config.get("model", config)
    return VisionTransformer(model_cfg)
