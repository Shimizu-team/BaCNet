#!/usr/bin/env python3
"""Build the fixed 30-compound ANNalog/BaCNet demonstration assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch


EMBEDDINGS = {
    "mf_embedding.pt": ("morgan_fingerprint.pt", 1024),
    "cc_embedding.pt": ("chemical_checker.pt", 1280),
    "chembert_embedding.pt": ("chemberta-2.pt", 384),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--reference-scores", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--reference-ecdf", type=Path, required=True)
    parser.add_argument("--reference-protein", type=Path, required=True)
    parser.add_argument(
        "--trust-legacy-pickle",
        action="store_true",
        help="Allow loading trusted legacy .pt files that contain NumPy arrays.",
    )
    return parser.parse_args()


def load_legacy(path: Path, trust_legacy_pickle: bool) -> dict:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        if not trust_legacy_pickle:
            raise RuntimeError(
                f"{path} requires legacy pickle loading; rerun only for a trusted source "
                "with --trust-legacy-pickle"
            )
        value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a dictionary")
    return value


def normalize_embeddings(
    raw: dict, compound_ids: list[str], expected_dim: int, source: Path
) -> dict[str, torch.Tensor]:
    by_string_id = {str(key): value for key, value in raw.items()}
    missing = [compound_id for compound_id in compound_ids if compound_id not in by_string_id]
    if missing:
        raise ValueError(f"{source} is missing Compound_ID values: {missing}")

    result: dict[str, torch.Tensor] = {}
    for compound_id in compound_ids:
        vector = torch.as_tensor(by_string_id[compound_id], dtype=torch.float32).flatten().cpu()
        if vector.numel() != expected_dim:
            raise ValueError(
                f"{source}: Compound_ID {compound_id} has {vector.numel()} dimensions; "
                f"expected {expected_dim}"
            )
        if not torch.isfinite(vector).all():
            raise ValueError(f"{source}: Compound_ID {compound_id} contains NaN or Inf")
        result[compound_id] = vector
    return result


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")

    filtered_path = args.source_dir / "filtered_data.csv"
    annotations_path = args.source_dir / "data_with_filters.csv"
    filtered = pd.read_csv(filtered_path, dtype={"Compound_ID": str})
    annotations = pd.read_csv(annotations_path, dtype={"Compound_ID": str})

    required_filtered = {"Compound_ID", "SMILES", "Fraggle_similarity", "Morgan_similarity"}
    required_annotations = {"Compound_ID", "SMILES", "Score", "PAINS", "SA_score", "QED"}
    if missing := required_filtered - set(filtered.columns):
        raise ValueError(f"{filtered_path} is missing columns: {sorted(missing)}")
    if missing := required_annotations - set(annotations.columns):
        raise ValueError(f"{annotations_path} is missing columns: {sorted(missing)}")
    if filtered["Compound_ID"].duplicated().any():
        raise ValueError("filtered_data.csv contains duplicate Compound_ID values")

    selected = filtered.iloc[: args.count].copy()
    if len(selected) != args.count:
        raise ValueError(f"Requested {args.count} compounds, but only {len(filtered)} are available")
    selected.insert(0, "source_row", range(1, len(selected) + 1))
    selected.insert(1, "generation_class", "medium")

    annotation_columns = ["Compound_ID", "SMILES", "Score", "PAINS", "SA_score", "QED"]
    merged = selected.merge(
        annotations[annotation_columns],
        on="Compound_ID",
        how="left",
        validate="one_to_one",
        suffixes=("", "_annotation"),
    )
    if merged["SMILES_annotation"].isna().any():
        raise ValueError("Missing filter annotations for one or more selected compounds")
    if not (merged["SMILES"] == merged["SMILES_annotation"]).all():
        raise ValueError("SMILES differ between filtered_data.csv and data_with_filters.csv")
    merged = merged.drop(columns="SMILES_annotation").rename(columns={"Score": "ANNalog_score"})
    if merged["PAINS"].astype(str).str.lower().isin({"true", "1"}).any():
        raise ValueError("A selected compound is PAINS-positive")
    if not ((merged["SA_score"] <= 5.0) & (merged["QED"] >= 0.5)).all():
        raise ValueError("A selected compound does not satisfy SA <= 5 and QED >= 0.5")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir = args.output_dir / "embeddings"
    expected_dir = args.output_dir / "expected"
    embedding_dir.mkdir(exist_ok=True)
    expected_dir.mkdir(exist_ok=True)

    compounds_path = args.output_dir / "compounds.csv"
    merged.to_csv(compounds_path, index=False)
    compound_ids = merged["Compound_ID"].tolist()

    output_files = [compounds_path]
    embedding_metadata: dict[str, dict[str, object]] = {}
    for source_name, (output_name, expected_dim) in EMBEDDINGS.items():
        source_path = args.source_dir / source_name
        raw = load_legacy(source_path, args.trust_legacy_pickle)
        normalized = normalize_embeddings(raw, compound_ids, expected_dim, source_path)
        output_path = embedding_dir / output_name
        torch.save(normalized, output_path)
        reloaded = torch.load(output_path, map_location="cpu", weights_only=True)
        for compound_id in compound_ids:
            if not torch.equal(normalized[compound_id], reloaded[compound_id]):
                raise AssertionError(f"Round-trip mismatch for {output_name}: {compound_id}")
        output_files.append(output_path)
        embedding_metadata[output_name] = {
            "dimension": expected_dim,
            "count": len(normalized),
            "sha256": sha256(output_path),
        }

    reference = pd.read_csv(args.reference_scores, dtype={"Compound_ID": str})
    if not {"Compound_ID", "CPI_score"}.issubset(reference.columns):
        raise ValueError("Reference score file must contain Compound_ID and CPI_score")
    if reference["Compound_ID"].duplicated().any():
        raise ValueError("Reference score file contains duplicate Compound_ID values")
    expected = merged[["source_row", "Compound_ID"]].merge(
        reference[["Compound_ID", "CPI_score"]],
        on="Compound_ID",
        how="left",
        validate="one_to_one",
    )
    if expected["CPI_score"].isna().any():
        missing = expected.loc[expected["CPI_score"].isna(), "Compound_ID"].tolist()
        raise ValueError(f"Reference scores are missing Compound_ID values: {missing}")
    expected_path = expected_dir / "P02918_screening_score.csv"
    expected.to_csv(expected_path, index=False)
    output_files.append(expected_path)

    source_files = [filtered_path, annotations_path, args.reference_scores]
    reference_files = [args.reference_model, args.reference_ecdf, args.reference_protein]
    manifest = {
        "description": "First 30 filter-passing compounds from ANNalog medium generation 1",
        "selection_rule": "first 30 rows of filtered_data.csv in source order",
        "compound_count": len(compound_ids),
        "compound_ids": compound_ids,
        "score_reference": (
            "Scores produced while evaluating the complete 694-compound medium library "
            "with the E. coli-held-out checkpoint"
        ),
        "sources": {path.name: sha256(path) for path in source_files},
        "references": {path.name: sha256(path) for path in reference_files},
        "embeddings": embedding_metadata,
        "outputs": {path.relative_to(args.output_dir).as_posix(): sha256(path) for path in output_files},
    }
    manifest_path = args.output_dir / "demo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(compound_ids)} compounds in {args.output_dir}")


if __name__ == "__main__":
    main()
