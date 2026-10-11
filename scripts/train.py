"""Training loop and experiment runner."""
from __future__ import annotations

import copy
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import get_cifar10_loaders
from models.model import build_vit_from_config
from utils.metrics import AverageMeter, accuracy, attention_matrix_size, count_parameters, format_large_number
from utils.visualization import plot_training_curves


def load_yaml(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def deep_update(base: Dict, updates: Dict) -> Dict:
    """Recursively update nested dictionaries."""
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device(config: Dict) -> torch.device:
    requested = str(config.get("training", {}).get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_optimizer(model: nn.Module, config: Dict):
    training_cfg = config.get("training", {})
    optimizer_name = str(training_cfg.get("optimizer", "adamw")).lower()
    lr = float(training_cfg.get("learning_rate", 3e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.05))

    if optimizer_name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_clip: float = 0.0):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy(logits.detach(), targets), batch_size)

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy(logits, targets), batch_size)

    return loss_meter.avg, acc_meter.avg


def summarize_experiment(config: Dict, model: nn.Module) -> Dict:
    model_cfg = config["model"]
    return {
        "parameters": count_parameters(model),
        "tokens": (model_cfg["image_size"] // model_cfg["patch_size"]) ** 2 + 1,
        "attention_matrix_elements_proxy": attention_matrix_size(
            image_size=model_cfg["image_size"],
            patch_size=model_cfg["patch_size"],
            depth=model_cfg["depth"],
            heads=model_cfg["num_heads"],
        ),
    }


def run_experiment(config: Dict, experiment_name: str = "baseline") -> Dict:
    training_cfg = config.get("training", {})
    output_cfg = config.get("outputs", {})

    set_seed(int(training_cfg.get("seed", 42)))
    device = get_device(config)

    train_loader, val_loader, test_loader = get_cifar10_loaders(config)
    model = build_vit_from_config(config).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(training_cfg.get("label_smoothing", 0.0)))
    optimizer = build_optimizer(model, config)
    epochs = int(training_cfg.get("epochs", 100))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(training_cfg.get("min_lr", 1e-6)))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(training_cfg.get("amp", True)) and device.type == "cuda")
    grad_clip = float(training_cfg.get("grad_clip", 1.0))

    log_dir = Path(output_cfg.get("log_dir", "outputs/logs"))
    plot_dir = Path(output_cfg.get("plot_dir", "outputs/plots"))
    ckpt_dir = Path(output_cfg.get("checkpoint_dir", "models/saved_models"))
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_experiment(config, model)
    print(
        f"Experiment={experiment_name} | device={device} | params={format_large_number(summary['parameters'])} | "
        f"tokens={summary['tokens']} | attention proxy={format_large_number(summary['attention_matrix_elements_proxy'])}"
    )

    history_path = log_dir / f"{experiment_name}_history.csv"
    best_path = ckpt_dir / f"{experiment_name}_best.pt"
    last_path = ckpt_dir / f"{experiment_name}_last.pt"

    history = []
    best_val_acc = -1.0
    start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, grad_clip)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.2f}% | "
            f"val loss {val_loss:.4f} acc {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "experiment_name": experiment_name,
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "summary": summary,
                },
                best_path,
            )

    # Save final checkpoint and history.
    torch.save(
        {
            "experiment_name": experiment_name,
            "model_state": model.state_dict(),
            "config": config,
            "epoch": epochs,
            "summary": summary,
        },
        last_path,
    )
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # Evaluate test using the best validation checkpoint.
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    loss_plot, acc_plot = plot_training_curves(history_path, plot_dir)

    result = {
        "experiment_name": experiment_name,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history_csv": str(history_path),
        "loss_plot": str(loss_plot),
        "accuracy_plot": str(acc_plot),
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "elapsed_seconds": time.time() - start,
        **summary,
    }
    result_path = log_dir / f"{experiment_name}_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Test loss {test_loss:.4f} | test acc {test_acc:.2f}%")
    print(f"Saved: {history_path}, {loss_plot}, {acc_plot}, {best_path}")
    return result


def run_named_experiment(config_path: str | Path, experiment_name: str) -> Dict:
    config = load_yaml(config_path)
    experiments = config.get("experiments", {})
    if experiment_name != "baseline":
        if experiment_name not in experiments:
            raise KeyError(f"Experiment '{experiment_name}' not found in config experiments")
        config = deep_update(config, experiments[experiment_name])
    return run_experiment(config, experiment_name)


def run_all_experiments(config_path: str | Path) -> Dict[str, Dict]:
    base_config = load_yaml(config_path)
    #results = {"baseline": run_experiment(base_config, "baseline")}
    results = {}
    for name, overrides in base_config.get("experiments", {}).items():
        config = deep_update(base_config, overrides)
        results[name] = run_experiment(config, name)
    return results
