#!/usr/bin/env python3
"""Command-line entry point for BaCNet training."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from model import BACNET_INPUT_DIM, BaCNet
from training import train_model
from training_data import (
    create_dataloaders,
    load_training_inputs,
    make_synthetic_inputs,
    validate_training_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BaCNet using precomputed target_bct values.")
    parser.add_argument("--config", default="configs/train_example.yaml")
    parser.add_argument("--links")
    parser.add_argument("--protein-embeddings")
    parser.add_argument("--morgan-embeddings")
    parser.add_argument("--cc-embeddings")
    parser.add_argument("--chemberta-embeddings")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--synthetic", action="store_true", help="Use deterministic test-only synthetic data.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise ImportError("PyYAML is required to read training configuration files.") from error
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Training config must contain a YAML mapping.")
    return config


def fix_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(cli_value: str | None, config: dict[str, Any], key: str) -> str:
    value = cli_value or config.get("data", {}).get(key)
    if not value or str(value).upper() == "PLACEHOLDER":
        raise ValueError(f"A real path is required for data.{key}; use --synthetic for a smoke test.")
    return str(value)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training_config = config.setdefault("training", {})
    model_config = config.setdefault("model", {})
    seed = int(training_config.get("seed", 123))
    fix_seeds(seed)

    output_dir = Path(args.output_dir or config.get("output_dir", "outputs/training_run")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"synthetic": bool(args.synthetic), "target": "target_bct"}
    if args.synthetic:
        inputs = make_synthetic_inputs(seed)
    else:
        paths = {
            "links": resolve_path(args.links, config, "links"),
            "protein_embeddings": resolve_path(args.protein_embeddings, config, "protein_embeddings"),
            "morgan_embeddings": resolve_path(args.morgan_embeddings, config, "morgan_embeddings"),
            "chemical_checker_embeddings": resolve_path(args.cc_embeddings, config, "chemical_checker_embeddings"),
            "chemberta_embeddings": resolve_path(args.chemberta_embeddings, config, "chemberta_embeddings"),
        }
        inputs = load_training_inputs(
            paths["links"],
            paths["protein_embeddings"],
            paths["morgan_embeddings"],
            paths["chemical_checker_embeddings"],
            paths["chemberta_embeddings"],
        )
        manifest["files"] = {
            name: {"path": str(Path(path).resolve()), "sha256": sha256(path)} for name, path in paths.items()
        }

    validation_report = validate_training_inputs(inputs)
    validation_report.to_csv(output_dir / "validation_report.csv", index=False)
    if not validation_report.empty:
        raise ValueError(
            f"Training data validation failed with {len(validation_report)} error(s). "
            f"See {output_dir / 'validation_report.csv'}"
        )

    split_summary = (
        inputs.links.assign(split=inputs.links["split"].astype(str).str.lower())
        .groupby("split", as_index=False)
        .agg(pairs=("target_bct", "size"), proteins=("protein_id", "nunique"), compounds=("compound_id", "nunique"))
    )
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    manifest["rows"] = len(inputs.links)
    manifest["embedding_dimensions"] = {"protein": 5120, "morgan": 1024, "chemical_checker": 1280, "chemberta": 384}
    with (output_dir / "data_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    try:
        import yaml
        with (output_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
    except ImportError:
        pass

    batch_size = int(training_config.get("batch_size", 32))
    loaders = create_dataloaders(
        inputs,
        batch_size=batch_size,
        num_workers=int(training_config.get("num_workers", 0)),
        seed=seed,
    )
    model = BaCNet(
        input_dim=BACNET_INPUT_DIM,
        num_features=list(model_config.get("hidden_dimensions", [1024, 256, 32])),
        dropout_rate=float(model_config.get("dropout", 0.01)),
    )
    optimizer_name = str(training_config.get("optimizer", "AdamW"))
    optimizer_class = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}.get(optimizer_name)
    if optimizer_class is None:
        raise ValueError("training.optimizer must be Adam or AdamW.")
    optimizer = optimizer_class(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-4)),
        weight_decay=float(training_config.get("weight_decay", 1e-3)),
    )

    try:
        from accelerate import Accelerator
    except ImportError as error:
        raise ImportError("accelerate is required for training. Install it in the training environment.") from error
    device_setting = str(training_config.get("device", "auto"))
    if device_setting not in {"auto", "cpu"}:
        raise ValueError("training.device must be 'auto' or 'cpu'. Use CUDA_VISIBLE_DEVICES for GPU selection.")
    accelerator = Accelerator(cpu=device_setting == "cpu")
    train_model(
        model=model,
        loaders=loaders,
        optimizer=optimizer,
        accelerator=accelerator,
        output_dir=output_dir,
        config=config,
        epochs=int(training_config.get("epochs", 200)),
        patience=int(training_config.get("patience", 10)),
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
