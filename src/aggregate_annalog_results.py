"""Aggregate the two-stage ANNalog/BaCNet workflow into a traceable CSV.

The input manifest describes only runs that contribute to the reported Figure 4
workflow. Each run combines the pre-filter table (``data_with_filters.csv``)
with the corresponding BaCNet score table. Compounds that fail a chemical
filter remain in the output with an explicit failure reason and no fabricated
BaCNet score.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


RUN_COLUMNS = {
    "run_id",
    "iteration",
    "parent_id",
    "parent_smiles",
    "generation_class",
    "generation_method",
    "temperature",
    "seed",
    "filter_csv",
    "bacnet_csv",
    "included_in_figure",
}
FILTER_COLUMNS = {"Compound_ID", "SMILES", "Score", "PAINS", "SA_score", "QED"}
BACNET_COLUMNS = {"Compound_ID", "CPI_score"}
SELECTION_COLUMNS = {
    "source_run_id",
    "compound_id",
    "selected_parent_id",
    "selection_basis",
    "structure_method",
    "structure_assessment",
}

OUTPUT_COLUMNS = [
    "run_id",
    "iteration",
    "parent_id",
    "parent_smiles",
    "generation_class",
    "generation_method",
    "temperature",
    "seed",
    "compound_id",
    "smiles",
    "annalog_score",
    "pains",
    "sa_score",
    "qed",
    "filter_pass",
    "filter_failure_reason",
    "bacnet_score",
    "bacnet_rank",
    "selected_for_next_iteration",
    "selected_parent_id",
    "selection_basis",
    "structure_method",
    "structure_assessment",
    "source_filter_file",
    "source_bacnet_file",
]


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _parse_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Expected a boolean value, got: {value!r}")


def normalize_compound_id(value: object) -> str:
    """Normalize legacy IDs such as ``tensor(450)`` to ``450``."""
    text = str(value).strip()
    match = re.fullmatch(r"tensor\(([^)]+)\)", text)
    return match.group(1).strip() if match else text


def _filter_outcome(row: pd.Series) -> tuple[bool, str]:
    reasons: list[str] = []
    pains = str(row["PAINS"]).strip().lower()
    if pains not in {"true", "false"}:
        reasons.append("invalid_PAINS")
    elif pains == "true":
        reasons.append("PAINS")

    sa_score = pd.to_numeric(pd.Series([row["SA_score"]]), errors="coerce").iloc[0]
    qed = pd.to_numeric(pd.Series([row["QED"]]), errors="coerce").iloc[0]
    if pd.isna(sa_score):
        reasons.append("invalid_SA_score")
    elif float(sa_score) > 5.0:
        reasons.append("SA_score>5")
    if pd.isna(qed):
        reasons.append("invalid_QED")
    elif float(qed) < 0.5:
        reasons.append("QED<0.5")
    return not reasons, ";".join(reasons)


def _load_selection(selection_path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if selection_path is None:
        return {}
    selection = pd.read_csv(selection_path, dtype=str, keep_default_na=False)
    _require_columns(selection, SELECTION_COLUMNS, str(selection_path))
    records: dict[tuple[str, str], dict[str, str]] = {}
    for row in selection.to_dict("records"):
        key = (row["source_run_id"].strip(), normalize_compound_id(row["compound_id"]))
        if key in records:
            raise ValueError(f"Duplicate selection record: {key}")
        records[key] = row
    return records


def aggregate_annalog_results(
    manifest_path: str | Path,
    data_root: str | Path,
    output_path: str | Path,
    selection_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> pd.DataFrame:
    """Aggregate manifest-defined runs and write a stable, long-format CSV."""
    manifest_file = Path(manifest_path)
    root = Path(data_root)
    output_file = Path(output_path)
    manifest = pd.read_csv(manifest_file, dtype=str, keep_default_na=False)
    _require_columns(manifest, RUN_COLUMNS, str(manifest_file))
    if manifest["run_id"].duplicated().any():
        duplicates = manifest.loc[manifest["run_id"].duplicated(keep=False), "run_id"].tolist()
        raise ValueError(f"Duplicate run_id values: {duplicates}")

    selection_file = Path(selection_path) if selection_path else None
    selected = _load_selection(selection_file)
    outputs: list[pd.DataFrame] = []

    for run in manifest.to_dict("records"):
        if not _parse_bool(run["included_in_figure"]):
            continue
        run_id = run["run_id"].strip()
        filter_path = root / run["filter_csv"]
        bacnet_path = root / run["bacnet_csv"]
        if not filter_path.is_file():
            raise FileNotFoundError(f"Filter table not found for {run_id}: {filter_path}")
        if not bacnet_path.is_file():
            raise FileNotFoundError(f"BaCNet table not found for {run_id}: {bacnet_path}")

        filters = pd.read_csv(filter_path, dtype={"Compound_ID": str})
        scores = pd.read_csv(bacnet_path, dtype={"Compound_ID": str})
        _require_columns(filters, FILTER_COLUMNS, str(filter_path))
        _require_columns(scores, BACNET_COLUMNS, str(bacnet_path))

        filters = filters.copy()
        filters["compound_id"] = filters["Compound_ID"].map(normalize_compound_id)
        if filters["compound_id"].duplicated().any():
            raise ValueError(f"Duplicate Compound_ID in {filter_path}")

        scores = scores.copy()
        scores["compound_id"] = scores["Compound_ID"].map(normalize_compound_id)
        scores["bacnet_score"] = pd.to_numeric(scores["CPI_score"], errors="raise")
        scores = scores.sort_values("bacnet_score", ascending=False, kind="stable")
        scores["bacnet_rank"] = np.arange(1, len(scores) + 1)
        if scores["compound_id"].duplicated().any():
            raise ValueError(f"Duplicate Compound_ID in {bacnet_path}")

        merged = filters.merge(
            scores[["compound_id", "bacnet_score", "bacnet_rank"]],
            on="compound_id",
            how="left",
            validate="one_to_one",
        )
        outcomes = merged.apply(_filter_outcome, axis=1, result_type="expand")
        merged["filter_pass"] = outcomes[0].astype(bool)
        merged["filter_failure_reason"] = outcomes[1]
        scored_failures = merged[~merged["filter_pass"] & merged["bacnet_score"].notna()]
        if not scored_failures.empty:
            ids = scored_failures["compound_id"].tolist()[:10]
            raise ValueError(f"Filtered-out compounds have BaCNet scores in {run_id}: {ids}")

        result = pd.DataFrame(
            {
                "run_id": run_id,
                "iteration": int(run["iteration"]),
                "parent_id": run["parent_id"],
                "parent_smiles": run["parent_smiles"],
                "generation_class": run["generation_class"],
                "generation_method": run["generation_method"],
                "temperature": float(run["temperature"]),
                "seed": int(run["seed"]),
                "compound_id": merged["compound_id"],
                "smiles": merged["SMILES"],
                "annalog_score": pd.to_numeric(merged["Score"], errors="coerce"),
                "pains": merged["PAINS"].astype(str).str.lower().map({"true": True, "false": False}),
                "sa_score": pd.to_numeric(merged["SA_score"], errors="coerce"),
                "qed": pd.to_numeric(merged["QED"], errors="coerce"),
                "filter_pass": merged["filter_pass"],
                "filter_failure_reason": merged["filter_failure_reason"],
                "bacnet_score": merged["bacnet_score"],
                "bacnet_rank": merged["bacnet_rank"].astype("Int64"),
                "source_filter_file": run["filter_csv"],
                "source_bacnet_file": run["bacnet_csv"],
            }
        )
        selection_records = [selected.get((run_id, compound_id), {}) for compound_id in result["compound_id"]]
        result["selected_for_next_iteration"] = [bool(record) for record in selection_records]
        for column in (
            "selected_parent_id",
            "selection_basis",
            "structure_method",
            "structure_assessment",
        ):
            result[column] = [record.get(column, "") for record in selection_records]
        outputs.append(result)

    if not outputs:
        raise ValueError("The manifest does not contain any included runs.")
    combined = pd.concat(outputs, ignore_index=True)[OUTPUT_COLUMNS]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_file, index=False)
    if summary_path is not None:
        summary_file = Path(summary_path)
        summary = (
            combined.groupby(
                [
                    "run_id",
                    "iteration",
                    "parent_id",
                    "generation_class",
                    "generation_method",
                    "temperature",
                    "seed",
                ],
                sort=False,
                dropna=False,
            )
            .agg(
                generated_count=("compound_id", "size"),
                filter_pass_count=("filter_pass", "sum"),
                bacnet_scored_count=("bacnet_score", "count"),
                selected_parent_count=("selected_for_next_iteration", "sum"),
                maximum_bacnet_score=("bacnet_score", "max"),
            )
            .reset_index()
        )
        summary["filter_failure_count"] = summary["generated_count"] - summary["filter_pass_count"]
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_file, index=False)
    return combined


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate ANNalog filtering and BaCNet scoring results for Figure 4."
    )
    parser.add_argument("--manifest", required=True, help="Run manifest CSV.")
    parser.add_argument("--data-root", required=True, help="Root directory for manifest-relative input paths.")
    parser.add_argument("--selection", help="Optional selection manifest CSV.")
    parser.add_argument("--output", required=True, help="Output long-format CSV.")
    parser.add_argument("--summary-output", help="Optional per-run summary CSV.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    combined = aggregate_annalog_results(
        manifest_path=args.manifest,
        data_root=args.data_root,
        output_path=args.output,
        selection_path=args.selection,
        summary_path=args.summary_output,
    )
    print(f"Wrote {len(combined):,} rows across {combined['run_id'].nunique()} runs to {args.output}")


if __name__ == "__main__":
    main()
