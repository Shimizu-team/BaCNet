#!/usr/bin/env python3
"""Run first-generation ANNalog generation and BaCNet ranking end to end."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate medium/far ANNalog compounds, embed them, and rank them with BaCNet."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse completed stage outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    value = mapping.get(key)
    if value is None or str(value).strip() == "" or "PLACEHOLDER" in str(value).upper():
        raise ValueError(f"Missing required configuration: {section}.{key}")
    return value


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _conda_command(conda: str, environment: str, script: Path, arguments: list[str]) -> list[str]:
    return [conda, "run", "--no-capture-output", "-n", environment, "python", str(script), *arguments]


class CommandRunner:
    def __init__(self, log_dir: Path, dry_run: bool) -> None:
        self.log_dir = log_dir
        self.dry_run = dry_run
        self.commands: list[dict[str, Any]] = []
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run(self, stage: str, command: list[str], expected: Path | None = None, resume: bool = False) -> None:
        record = {"stage": stage, "command": command, "status": "planned"}
        self.commands.append(record)
        printable = shlex.join(command)
        if resume and expected is not None and expected.exists():
            record["status"] = "reused"
            print(f"[reuse:{stage}] {expected}")
            return
        print(f"[run:{stage}] {printable}")
        if self.dry_run:
            record["status"] = "dry-run"
            return
        log_path = self.log_dir / f"{stage}.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            return_code = process.wait()
        record["return_code"] = return_code
        record["log"] = str(log_path.resolve())
        if return_code != 0:
            record["status"] = "failed"
            raise subprocess.CalledProcessError(return_code, command)
        if expected is not None and not expected.exists():
            record["status"] = "failed"
            raise FileNotFoundError(f"Stage {stage} did not create {expected}")
        record["status"] = "completed"


def merge_rankings(compounds_path: Path, score_path: Path, output_path: Path) -> int:
    with compounds_path.open(newline="", encoding="utf-8") as handle:
        compounds = list(csv.DictReader(handle))
    with score_path.open(newline="", encoding="utf-8") as handle:
        scores = list(csv.DictReader(handle))
    if not scores or {"Compound_ID", "CPI_score"} - set(scores[0]):
        raise ValueError(f"Invalid BaCNet score file: {score_path}")
    score_by_id: dict[str, float] = {}
    for row in scores:
        compound_id = str(row["Compound_ID"])
        if compound_id in score_by_id:
            raise ValueError(f"Duplicate Compound_ID in score file: {compound_id}")
        score_by_id[compound_id] = float(row["CPI_score"])
    merged = []
    for row in compounds:
        compound_id = str(row["Compound_ID"])
        if compound_id in score_by_id:
            merged.append({**row, "CPI_score": score_by_id[compound_id]})
    merged.sort(key=lambda row: float(row["CPI_score"]), reverse=True)
    for rank, row in enumerate(merged, 1):
        row["BaCNet_rank"] = rank
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(compounds[0].keys()) + ["CPI_score", "BaCNet_rank"] if compounds else []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)
    return len(merged)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")

    conda = str(config.get("conda_executable", "conda"))
    if not args.dry_run and shutil.which(conda) is None and not Path(conda).is_file():
        raise FileNotFoundError(f"Conda executable not found: {conda}")

    environments = config.get("environments", {})
    annalog_env = str(_required(environments, "annalog", "environments"))
    morgan_env = str(_required(environments, "morgan", "environments"))
    chemberta_env = str(_required(environments, "chemberta", "environments"))
    cc_env = str(_required(environments, "chemical_checker", "environments"))
    bacnet_env = str(_required(environments, "bacnet", "environments"))

    annalog = config.get("annalog", {})
    embeddings = config.get("embeddings", {})
    bacnet = config.get("bacnet", {})
    input_smiles = str(_required(config, "input_smiles", "root"))
    output_dir = _path(_required(config, "output_dir", "root"))
    vocab_path = _path(_required(annalog, "vocab_path", "annalog"))
    checkpoint_path = _path(_required(annalog, "checkpoint_path", "annalog"))
    seed = int(annalog.get("seed", 42))
    generation_number = int(annalog.get("generation_number", 1000))
    cc_param_dir = _path(_required(embeddings, "cc_param_dir", "embeddings"))
    protein_embedding = _path(_required(bacnet, "protein_embedding", "bacnet"))
    model_path = _path(bacnet.get("model", "models/checkpoint.pt"))
    ecdf_path = _path(bacnet.get("ecdf", "models/ecdf_bacnet_v1.npz"))

    resolved_config = output_dir / "config.resolved.yaml"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume and not args.dry_run:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --resume or choose a new output_dir."
        )
    if args.resume and resolved_config.is_file():
        previous = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError(
                "The configuration differs from the existing run. Choose a new output_dir instead of --resume."
            )

    required_files = [vocab_path, checkpoint_path, protein_embedding, model_path, ecdf_path]
    if not args.dry_run:
        for path in required_files:
            if not path.is_file():
                raise FileNotFoundError(path)
        if not cc_param_dir.is_dir():
            raise NotADirectoryError(cc_param_dir)

    generated_dir = output_dir / "generated"
    prepared_dir = output_dir / "prepared"
    stage_dir = output_dir / "stages"
    final_embeddings = output_dir / "chemical_embeddings"
    score_dir = output_dir / "scores"
    log_dir = output_dir / "logs"
    for directory in (generated_dir, prepared_dir, stage_dir, final_embeddings, score_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    resolved_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    runner = CommandRunner(log_dir, args.dry_run)
    generation_script = REPOSITORY_ROOT / "src" / "run_annalog_generation.py"
    medium_csv = generated_dir / "medium.csv"
    far_csv = generated_dir / "far.csv"
    for generation_class, temperature, destination in (
        ("medium", float(annalog.get("medium_temperature", 1.2)), medium_csv),
        ("far", float(annalog.get("far_temperature", 1.5)), far_csv),
    ):
        command = _conda_command(
            conda,
            annalog_env,
            generation_script,
            [
                "--input-smiles", input_smiles,
                "--generation-class", generation_class,
                "--generation-method", "sampling",
                "--temperature", str(temperature),
                "--generation-number", str(generation_number),
                "--seed", str(seed),
                "--vocab-path", str(vocab_path),
                "--checkpoint-path", str(checkpoint_path),
                "--output-csv", str(destination),
            ],
        )
        runner.run(f"generate_{generation_class}", command, destination, args.resume)

    prepared_csv = prepared_dir / "annalog_compounds.csv"
    rejected_csv = prepared_dir / "invalid_generated_smiles.csv"
    prepare_command = _conda_command(
        conda,
        morgan_env,
        REPOSITORY_ROOT / "src" / "prepare_annalog_compounds.py",
        [
            "--medium", str(medium_csv),
            "--far", str(far_csv),
            "--output", str(prepared_csv),
            "--rejected", str(rejected_csv),
            "--manifest", str(prepared_dir / "preparation_manifest.json"),
        ],
    )
    runner.run("prepare_compounds", prepare_command, prepared_csv, args.resume)

    method_specs = [
        ("morgan", morgan_env, prepared_csv, [], "morgan_fingerprint.pt"),
        (
            "chemical_checker",
            cc_env,
            stage_dir / "morgan" / "data_with_filters.csv",
            ["--cc-param-dir", str(cc_param_dir)],
            "chemical_checker.pt",
        ),
        (
            "chemberta",
            chemberta_env,
            stage_dir / "morgan" / "data_with_filters.csv",
            [
                "--device", str(embeddings.get("chemberta_device", "auto")),
                "--chemberta-batch-size", str(int(embeddings.get("chemberta_batch_size", 16))),
            ],
            "chemberta-2.pt",
        ),
    ]
    method_outputs: dict[str, Path] = {}
    failure_csvs: list[Path] = []
    for method, environment, input_csv, extra, filename in method_specs:
        method_dir = stage_dir / method
        expected = method_dir / "Chemical_embeddings" / filename
        command = _conda_command(
            conda,
            environment,
            REPOSITORY_ROOT / "src" / "chemical_embedding.py",
            [
                "--input_csv", str(input_csv),
                "--output-dir", str(method_dir),
                "--methods", method,
                "--resume",
                *extra,
            ],
        )
        runner.run(f"embed_{method}", command, expected, args.resume)
        method_outputs[method] = expected
        failure_csvs.append(method_dir / "embedding_failures.csv")

    scoring_compounds = final_embeddings / "scoring_compounds.csv"
    assemble_args = [
        "--compounds", str(stage_dir / "morgan" / "data_with_filters.csv"),
        "--morgan", str(method_outputs["morgan"]),
        "--chemical-checker", str(method_outputs["chemical_checker"]),
        "--chemberta", str(method_outputs["chemberta"]),
        "--output-dir", str(final_embeddings),
    ]
    for failure_csv in failure_csvs:
        assemble_args.extend(["--failure-csv", str(failure_csv)])
    assemble_command = _conda_command(
        conda,
        bacnet_env,
        REPOSITORY_ROOT / "src" / "assemble_annalog_embeddings.py",
        assemble_args,
    )
    runner.run("assemble_embeddings", assemble_command, scoring_compounds, args.resume)

    inference_command = _conda_command(
        conda,
        bacnet_env,
        REPOSITORY_ROOT / "src" / "search_drug.py",
        [
            "--model", str(model_path),
            "--ecdf", str(ecdf_path),
            "--protein", str(protein_embedding),
            "--chemical", str(final_embeddings),
            "--output", str(score_dir),
        ],
    )
    existing_scores = sorted(score_dir.glob("*_screening_score.csv"))
    expected_score = existing_scores[0] if len(existing_scores) == 1 else None
    runner.run("bacnet_inference", inference_command, expected_score, args.resume and expected_score is not None)

    ranked_path = output_dir / "ranked_candidates.csv"
    ranked_count = 0
    if not args.dry_run:
        score_files = sorted(score_dir.glob("*_screening_score.csv"))
        if len(score_files) != 1:
            raise RuntimeError(f"Expected exactly one protein score file in {score_dir}, found {len(score_files)}")
        ranked_count = merge_rankings(scoring_compounds, score_files[0], ranked_path)

    manifest = {
        "status": "dry-run" if args.dry_run else "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "seed": seed,
        "input_files": {
            "annalog_vocabulary": {
                "path": str(vocab_path.resolve()),
                "sha256": sha256(vocab_path) if vocab_path.is_file() else None,
            },
            "annalog_checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": sha256(checkpoint_path) if checkpoint_path.is_file() else None,
            },
            "protein_embedding": {
                "path": str(protein_embedding.resolve()),
                "sha256": sha256(protein_embedding) if protein_embedding.is_file() else None,
            },
            "bacnet_model": {
                "path": str(model_path.resolve()),
                "sha256": sha256(model_path) if model_path.is_file() else None,
            },
            "ecdf": {
                "path": str(ecdf_path.resolve()),
                "sha256": sha256(ecdf_path) if ecdf_path.is_file() else None,
            },
            "chemical_checker_parameters": {"path": str(cc_param_dir.resolve())},
        },
        "commands": runner.commands,
        "ranked_candidates": ranked_count,
        "outputs": {
            "ranked_candidates": str(ranked_path.resolve()),
            "filter_annotations": str((stage_dir / "morgan" / "filter_annotations.csv").resolve()),
            "embedding_failures": str((final_embeddings / "embedding_failures.csv").resolve()),
            "chemical_embeddings": str(final_embeddings.resolve()),
            "scores": str(score_dir.resolve()),
        },
    }
    (output_dir / "workflow_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Workflow {'planned' if args.dry_run else 'completed'}: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
