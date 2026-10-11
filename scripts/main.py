"""Main entry point for training, sweeps, evaluation, and attention visualization."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import (
    evaluate_checkpoint,
    save_pretrained_vit_attention_maps,
    save_scratch_attention_maps,
)
from scripts.train import run_all_experiments, run_named_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ViT from scratch on CIFAR-10")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "sweep", "evaluate", "attention_scratch", "attention_pretrained"],
        help="What to run",
    )
    parser.add_argument("--experiment", type=str, default="baseline", help="Experiment name in config")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path for evaluate/attention_scratch")
    parser.add_argument("--head-index", type=int, default=0, help="Attention head index to plot")
    parser.add_argument("--num-images", type=int, default=2, help="Number of CIFAR-10 test images for scratch attention")
    parser.add_argument(
        "--use-logits",
        action="store_true",
        help="Plot raw scaled QK^T similarity instead of softmax probabilities",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "train":
        result = run_named_experiment(args.config, args.experiment)
        print(json.dumps(result, indent=2))
        return

    if args.mode == "sweep":
        results = run_all_experiments(args.config)
        print(json.dumps(results, indent=2))
        return

    if args.mode == "evaluate":
        if not args.checkpoint:
            args.checkpoint = f"models/saved_models/{args.experiment}_best.pt"
        result = evaluate_checkpoint(args.checkpoint)
        print(json.dumps(result, indent=2))
        return

    if args.mode == "attention_scratch":
        if not args.checkpoint:
            args.checkpoint = f"models/saved_models/{args.experiment}_best.pt"
        save_scratch_attention_maps(
            checkpoint_path=args.checkpoint,
            head_index=args.head_index,
            num_images=args.num_images,
            use_logits=args.use_logits,
        )
        return

    if args.mode == "attention_pretrained":
        save_pretrained_vit_attention_maps(
            config_path=args.config,
            head_index=args.head_index,
            use_logits=args.use_logits,
        )
        return


if __name__ == "__main__":
    main()
