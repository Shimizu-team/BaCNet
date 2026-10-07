#!/usr/bin/env python3
"""Validate separately generated embeddings and assemble a BaCNet library."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch


EXPECTED = {
    "morgan": (1024, "morgan_fingerprint.pt"),
    "chemical_checker": (1280, "chemical_checker.pt"),
    "chemberta": (384, "chemberta-2.pt"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble ANNalog chemical embeddings for BaCNet.")
    parser.add_argument("--compounds", type=Path, required=True)
    parser.add_argument("--morgan", type=Path, required=True)
    parser.add_argument("--chemical-checker", type=Path, required=True)
    parser.add_argument("--chemberta", type=Path, required=True)
    parser.add_argument("--failure-csv", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_valid(path: Path, method: str, expected_dim: int) -> tuple[dict[str, torch.Tensor], list[dict[str, str]]]:
    raw = torch.load(path, map_location=torch.device("cpu"))
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a dictionary")
    valid: dict[str, torch.Tensor] = {}
    failures: list[dict[str, str]] = []
    for identifier, value in raw.items():
        compound_id = str(identifier)
        try:
            vector = torch.as_tensor(value, dtype=torch.float32).flatten().cpu()
            if vector.numel() != expected_dim:
                raise ValueError(f"expected {expected_dim} dimensions, got {vector.numel()}")
            if not torch.isfinite(vector).all():
                raise ValueError("embedding contains NaN or Inf")
            valid[compound_id] = vector
        except Exception as error:
            failures.append(
                {
                    "Embedding_type": method,
                    "Compound_ID": compound_id,
                    "SMILES": "",
                    "Error_type": type(error).__name__,
                    "Error_message": str(error),
                }
            )
    return valid, failures


def main() -> None:
    args = parse_args()
    compounds = pd.read_csv(args.compounds, dtype={"Compound_ID": str})
    if "Compound_ID" not in compounds.columns:
        raise ValueError(f"{args.compounds} does not contain Compound_ID")
    if compounds["Compound_ID"].duplicated().any():
        raise ValueError("Compound_ID values must be unique")

    paths = {
        "morgan": args.morgan,
        "chemical_checker": args.chemical_checker,
        "chemberta": args.chemberta,
    }
    embeddings: dict[str, dict[str, torch.Tensor]] = {}
    failures: list[dict[str, str]] = []
    for method, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        embeddings[method], invalid = _load_valid(path, method, EXPECTED[method][0])
        failures.extend(invalid)

    requested = set(compounds["Compound_ID"].astype(str))
    common = requested.copy()
    for method, values in embeddings.items():
        missing = requested - set(values)
        for compound_id in sorted(missing):
            failures.append(
                {
                    "Embedding_type": method,
                    "Compound_ID": compound_id,
                    "SMILES": "",
                    "Error_type": "MissingEmbedding",
                    "Error_message": "No valid embedding was produced",
                }
            )
        common &= set(values)
    if not common:
        raise ValueError("No compound has all three valid embeddings")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, object]] = {}
    for method, values in embeddings.items():
        output = args.output_dir / EXPECTED[method][1]
        torch.save({key: values[key] for key in sorted(common)}, output)
        outputs[method] = {"path": str(output.resolve()), "sha256": sha256(output), "count": len(common)}

    for failure_csv in args.failure_csv:
        if failure_csv.is_file():
            frame = pd.read_csv(failure_csv, dtype=str).fillna("")
            failures.extend(frame.to_dict("records"))
    failure_columns = ["Embedding_type", "Compound_ID", "SMILES", "Error_type", "Error_message"]
    pd.DataFrame(failures).reindex(columns=failure_columns).drop_duplicates().to_csv(
        args.output_dir / "embedding_failures.csv", index=False
    )
    scoring = compounds[compounds["Compound_ID"].astype(str).isin(common)].copy()
    scoring.to_csv(args.output_dir / "scoring_compounds.csv", index=False)
    manifest = {
        "requested_compounds": len(requested),
        "scored_compounds": len(common),
        "excluded_compounds": len(requested - common),
        "dimensions": {method: value[0] for method, value in EXPECTED.items()},
        "outputs": outputs,
    }
    (args.output_dir / "embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Assembled {len(common)} compounds with all three embeddings")


if __name__ == "__main__":
    main()
