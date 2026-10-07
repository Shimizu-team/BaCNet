#!/usr/bin/env python3
"""Normalize and deduplicate ANNalog medium/far CSV outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ANNalog output for chemical embedding.")
    parser.add_argument("--medium", type=Path, required=True)
    parser.add_argument("--far", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def _stable_id(smiles: str) -> str:
    return "annalog_" + hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:16]


def prepare(medium: Path, far: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    input_counts: dict[str, int] = {}
    for generation_class, path in (("medium", medium), ("far", far)):
        frame = pd.read_csv(path)
        missing = {"SMILES", "Score"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        input_counts[generation_class] = len(frame)
        for zero_index, row in frame.iterrows():
            raw_smiles = str(row["SMILES"]).strip()
            source_row = int(zero_index) + 2
            molecule = Chem.MolFromSmiles(raw_smiles)
            if molecule is None:
                rejected.append(
                    {
                        "generation_class": generation_class,
                        "source_row": source_row,
                        "SMILES": raw_smiles,
                        "Error_type": "InvalidSMILES",
                        "Error_message": "RDKit could not parse the generated SMILES",
                    }
                )
                continue
            try:
                score = float(row["Score"])
            except (TypeError, ValueError) as error:
                rejected.append(
                    {
                        "generation_class": generation_class,
                        "source_row": source_row,
                        "SMILES": raw_smiles,
                        "Error_type": type(error).__name__,
                        "Error_message": f"Invalid ANNalog score: {row['Score']!r}",
                    }
                )
                continue
            accepted.append(
                {
                    "SMILES": Chem.MolToSmiles(molecule, canonical=True),
                    "ANNalog_score": score,
                    "generation_class": generation_class,
                    "source_row": source_row,
                }
            )

    frame = pd.DataFrame(accepted)
    if frame.empty:
        raise ValueError("ANNalog did not produce any valid SMILES")

    grouped_rows: list[dict[str, object]] = []
    for smiles, group in frame.groupby("SMILES", sort=True):
        best = group.sort_values("ANNalog_score", ascending=False, kind="stable").iloc[0]
        grouped_rows.append(
            {
                "Compound_ID": _stable_id(smiles),
                "SMILES": smiles,
                "ANNalog_score": float(best["ANNalog_score"]),
                "generation_class": str(best["generation_class"]),
                "source_generation_classes": ";".join(sorted(set(group["generation_class"].astype(str)))),
                "source_rows": ";".join(
                    f"{row.generation_class}:{int(row.source_row)}" for row in group.itertuples()
                ),
            }
        )
    prepared = pd.DataFrame(grouped_rows).sort_values("Compound_ID", kind="stable").reset_index(drop=True)
    rejected_frame = pd.DataFrame(
        rejected,
        columns=["generation_class", "source_row", "SMILES", "Error_type", "Error_message"],
    )
    counts = {
        "medium_input": input_counts.get("medium", 0),
        "far_input": input_counts.get("far", 0),
        "valid_before_deduplication": len(frame),
        "unique_valid_compounds": len(prepared),
        "rejected": len(rejected_frame),
        "duplicates_removed": len(frame) - len(prepared),
    }
    return prepared, rejected_frame, counts


def main() -> None:
    args = parse_args()
    for path in (args.medium, args.far):
        if not path.is_file():
            raise FileNotFoundError(path)
    prepared, rejected, counts = prepare(args.medium, args.far)
    for path in (args.output, args.rejected, args.manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output, index=False)
    rejected.to_csv(args.rejected, index=False)
    args.manifest.write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(prepared)} unique compounds: {args.output}")


if __name__ == "__main__":
    main()
