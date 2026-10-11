"""CIFAR-10 loading and preprocessing utilities."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def seed_worker(worker_id: int) -> None:
    """Make dataloader workers reproducible."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_transforms(config: Dict, train: bool) -> transforms.Compose:
    """Create train/eval transforms.

    For the 64x64 experiment, CIFAR-10 images are resized before patching.
    """
    dataset_cfg = config.get("dataset", {})
    train_cfg = config.get("training", {})
    image_size = int(config.get("model", {}).get("image_size", dataset_cfg.get("image_size", 32)))
    augment = bool(train_cfg.get("augment", True))
    randaugment = bool(train_cfg.get("randaugment", False))

    ops = []
    if image_size != 32:
        ops.append(transforms.Resize((image_size, image_size)))

    if train and augment:
        # RandomCrop after optional resize keeps the target image_size.
        padding = 4 if image_size <= 32 else 8
        ops.extend(
            [
                transforms.RandomCrop(image_size, padding=padding),
                transforms.RandomHorizontalFlip(),
            ]
        )
        if randaugment and hasattr(transforms, "RandAugment"):
            ops.append(transforms.RandAugment(num_ops=2, magnitude=9))

    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return transforms.Compose(ops)


def _split_train_val_indices(num_samples: int, val_fraction: float, seed: int) -> Tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator).tolist()
    val_size = int(num_samples * val_fraction)
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]
    return train_indices, val_indices


def get_cifar10_datasets(config: Dict):
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})

    root = Path(dataset_cfg.get("root", "data/dataset"))
    download = bool(dataset_cfg.get("download", True))
    val_fraction = float(dataset_cfg.get("val_fraction", 0.1))
    seed = int(training_cfg.get("seed", 42))

    train_transform = build_transforms(config, train=True)
    eval_transform = build_transforms(config, train=False)

    full_train_for_training = datasets.CIFAR10(root=root, train=True, download=download, transform=train_transform)
    full_train_for_validation = datasets.CIFAR10(root=root, train=True, download=download, transform=eval_transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=download, transform=eval_transform)

    train_indices, val_indices = _split_train_val_indices(len(full_train_for_training), val_fraction, seed)
    train_dataset = Subset(full_train_for_training, train_indices)
    val_dataset = Subset(full_train_for_validation, val_indices)
    return train_dataset, val_dataset, test_dataset


def get_cifar10_loaders(config: Dict):
    training_cfg = config.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 128))
    num_workers = int(training_cfg.get("num_workers", 2))
    seed = int(training_cfg.get("seed", 42))

    train_dataset, val_dataset, test_dataset = get_cifar10_datasets(config)
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    return train_loader, val_loader, test_loader


def get_raw_cifar10_test_dataset(config: Dict):
    """Untransformed test dataset used for pretrained ViT visualization."""
    dataset_cfg = config.get("dataset", {})
    root = Path(dataset_cfg.get("root", "data/dataset"))
    download = bool(dataset_cfg.get("download", True))
    return datasets.CIFAR10(root=root, train=False, download=download, transform=None)
