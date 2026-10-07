#!/usr/bin/env python3
"""Reproduce the BaCNet architecture-ablation experiment."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ablation_models import (
    ABLATION_DESCRIPTIONS,
    ABLATION_VARIANTS,
    MODALITY_DIMS,
    create_ablation_model,
)
from train import fix_seeds, load_config, sha256
from training import train_model
from training_data import (
    create_dataloaders,
    load_training_inputs_from_splits,
    make_synthetic_inputs,
    validate_training_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the BaCNet baseline and five alternative architectures on "
            "explicitly provided train/validation/test files."
        )
    )
    parser.add_argument("--config", default="configs/ablation.yaml")
    parser.add_argument("--train-links")
    parser.add_argument("--validation-links")
    parser.add_argument("--test-links")
    parser.add_argument("--protein-embeddings")
    parser.add_argument("--morgan-embeddings")
    parser.add_argument("--cc-embeddings")
    parser.add_argument("--chemberta-embeddings")
    parser.add_argument("--output-dir")
    parser.add_argument("--variants", nargs="+", choices=ABLATION_VARIANTS)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic inputs for a software smoke test only.",
    )
    return parser.parse_args()


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
    output_dir = Path(args.output_dir or config.get("output_dir", "outputs/ablation")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device_setting = str(training_config.get("device", "auto"))
    if device_setting not in {"auto", "cpu"}:
        raise ValueError(
            "training.device must be 'auto' or 'cpu'. "
            "Use CUDA_VISIBLE_DEVICES for GPU selection."
        )
    try:
        from accelerate import Accelerator
    except ImportError as error:
        raise ImportError("accelerate is required for ablation training.") from error
    accelerator = Accelerator(cpu=device_setting == "cpu")

    manifest: dict[str, Any] = {
        "synthetic": bool(args.synthetic),
        "target": "transformed_score",
        "model_seed": seed,
    }
    if args.synthetic:
        inputs = make_synthetic_inputs(seed)
    else:
        paths = {
            "train_links": resolve_path(args.train_links, config, "train_links"),
            "validation_links": resolve_path(args.validation_links, config, "validation_links"),
            "test_links": resolve_path(args.test_links, config, "test_links"),
            "protein_embeddings": resolve_path(args.protein_embeddings, config, "protein_embeddings"),
            "morgan_embeddings": resolve_path(args.morgan_embeddings, config, "morgan_embeddings"),
            "chemical_checker_embeddings": resolve_path(
                args.cc_embeddings, config, "chemical_checker_embeddings"
            ),
            "chemberta_embeddings": resolve_path(
                args.chemberta_embeddings, config, "chemberta_embeddings"
            ),
        }
        inputs = load_training_inputs_from_splits(
            paths["train_links"],
            paths["validation_links"],
            paths["test_links"],
            paths["protein_embeddings"],
            paths["morgan_embeddings"],
            paths["chemical_checker_embeddings"],
            paths["chemberta_embeddings"],
            column_names=config.get("data", {}).get("columns"),
        )
        manifest["files"] = {
            name: {"path": str(Path(path).resolve()), "sha256": sha256(path)}
            for name, path in paths.items()
        }

    validation_report = validate_training_inputs(inputs)
    if accelerator.is_main_process:
        validation_report.to_csv(output_dir / "validation_report.csv", index=False)
    if not validation_report.empty:
        raise ValueError(
            f"Ablation data validation failed with {len(validation_report)} error(s). "
            f"See {output_dir / 'validation_report.csv'}"
        )

    split_summary = (
        inputs.links.groupby("split", as_index=False)
        .agg(
            pairs=("transformed_score", "size"),
            proteins=("protein_id", "nunique"),
            compounds=("compound_id", "nunique"),
        )
    )
    split_counts = dict(zip(split_summary["split"], split_summary["pairs"]))
    if accelerator.is_main_process:
        split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    manifest["rows"] = int(len(inputs.links))
    manifest["embedding_dimensions"] = dict(MODALITY_DIMS)
    if accelerator.is_main_process:
        with (output_dir / "data_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        try:
            import yaml

            with (output_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
        except ImportError:
            pass
    accelerator.wait_for_everyone()

    optimizer_name = str(training_config.get("optimizer", "AdamW"))
    optimizer_class = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}.get(optimizer_name)
    if optimizer_class is None:
        raise ValueError("training.optimizer must be Adam or AdamW.")
    variants = args.variants or list(config.get("variants", ABLATION_VARIANTS))
    unknown_variants = sorted(set(variants) - set(ABLATION_VARIANTS))
    if unknown_variants:
        raise ValueError(f"Unknown ablation variant(s): {unknown_variants}")

    results: list[dict[str, Any]] = []
    for variant in variants:
        # Every architecture starts from the same documented single-run seed.
        fix_seeds(seed)
        loaders = create_dataloaders(
            inputs,
            batch_size=int(training_config.get("batch_size", 32)),
            num_workers=int(training_config.get("num_workers", 0)),
            seed=seed,
        )
        model = create_ablation_model(
            variant,
            projection_dim=int(model_config.get("projection_dimension", 256)),
            hidden_dims=list(model_config.get("hidden_dimensions", [1024, 256, 32])),
            dropout_rate=float(model_config.get("dropout", 0.01)),
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        optimizer = optimizer_class(
            model.parameters(),
            lr=float(training_config.get("learning_rate", 1e-4)),
            weight_decay=float(training_config.get("weight_decay", 1e-3)),
        )
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_config = deepcopy(config)
        variant_config["variant"] = variant
        history, test_metrics = train_model(
            model=model,
            loaders=loaders,
            optimizer=optimizer,
            accelerator=accelerator,
            output_dir=variant_dir,
            config=variant_config,
            epochs=int(training_config.get("epochs", 50)),
            patience=int(training_config.get("patience", 50)),
        )
        validation_rows = [row for row in history if row["split"] == "validation"]
        best_validation = min(validation_rows, key=lambda row: float(row["loss"]))
        result = {
            "variant": variant,
            "architecture": ABLATION_DESCRIPTIONS[variant],
            "model_seed": seed,
            "train_rows": int(split_counts["train"]),
            "validation_rows": int(split_counts["validation"]),
            "test_rows": int(split_counts["test"]),
            "parameters": parameter_count,
            "best_epoch": int(best_validation["epoch"]) + 1,
            "best_validation_loss": float(best_validation["loss"]),
            "test_loss": float(test_metrics["loss"]),
            "test_PCC": float(test_metrics["pcc"]),
            "test_SCC": float(test_metrics["scc"]),
            "test_r2": float(test_metrics["r2"]),
            "test_MSE": float(test_metrics["mse"]),
        }
        results.append(result)
        if accelerator.is_main_process:
            pd.DataFrame(results).to_csv(output_dir / "ablation_results.csv", index=False)
        accelerator.free_memory()


if __name__ == "__main__":
    main()
