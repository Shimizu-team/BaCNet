#!/usr/bin/env python3
"""Generate 5,120-dimensional ESM-2 protein embeddings from FASTA or CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch


ESM_LAYER = 48
ESM_DIMENSION = 5120
DEFAULT_MODEL_NAME = "esm2_t48_15B_UR50D"
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBXZUOJ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BaCNet-compatible ESM-2 embeddings.")
    parser.add_argument("--input", "-i", required=True, help="FASTA or CSV input path.")
    parser.add_argument("--output", "-o", required=True, help="Output .pt dictionary path.")
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model-path", help="Local ESM checkpoint path (recommended for reproducibility).")
    model_group.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Function name from esm.pretrained.")
    parser.add_argument("--backend", choices=["single", "fsdp"], default="single")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1022)
    parser.add_argument("--long-sequence-policy", choices=["error", "skip", "truncate"], default="error")
    parser.add_argument("--id-column", default="protein_id")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--failure-report", help="Failure CSV path; defaults beside --output.")
    parser.add_argument("--resume", action="store_true", help="Keep valid embeddings already present in --output.")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records = []
    identifier = None
    sequence: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    records.append((identifier, "".join(sequence)))
                identifier = line[1:].split()[0] if line[1:].strip() else ""
                sequence = []
            else:
                sequence.append(line)
    if identifier is not None:
        records.append((identifier, "".join(sequence)))
    return records


def read_sequences(path: Path, id_column: str, sequence_column: str) -> list[tuple[str, str]]:
    if path.suffix.lower() in {".fa", ".faa", ".fasta"}:
        return read_fasta(path)
    data = pd.read_csv(path)
    # Backward-compatible aliases for the original wrapper.
    if id_column not in data and id_column == "protein_id" and "Name" in data:
        id_column = "Name"
    if sequence_column not in data and sequence_column == "sequence" and "Sequence" in data:
        sequence_column = "Sequence"
    missing = [column for column in (id_column, sequence_column) if column not in data]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return [(str(identifier), str(sequence)) for identifier, sequence in zip(data[id_column], data[sequence_column])]


def validate_records(
    records: Iterable[tuple[str, str]], max_length: int, long_sequence_policy: str
) -> tuple[list[tuple[str, str]], list[dict[str, str]]]:
    valid = []
    failures = []
    seen = set()
    for identifier, raw_sequence in records:
        sequence = "".join(raw_sequence.split()).upper()
        error = None
        if not identifier:
            error = "empty protein identifier"
        elif identifier in seen:
            error = "duplicated protein identifier"
        elif not sequence:
            error = "empty sequence"
        elif set(sequence) - VALID_AMINO_ACIDS:
            error = f"unsupported amino acid symbols: {sorted(set(sequence) - VALID_AMINO_ACIDS)}"
        elif len(sequence) > max_length:
            if long_sequence_policy == "truncate":
                sequence = sequence[:max_length]
            else:
                error = f"sequence length {len(sequence)} exceeds maximum {max_length}"
        if error:
            failures.append({"protein_id": identifier, "stage": "validation", "error": error})
            if long_sequence_policy == "error" and "exceeds maximum" in error:
                raise ValueError(f"{identifier}: {error}")
            continue
        seen.add(identifier)
        valid.append((identifier, sequence))
    if not valid:
        raise ValueError("No valid protein sequences remain after validation.")
    return valid, failures


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(value)


def load_model(args: argparse.Namespace, device: torch.device):
    try:
        import esm
    except ImportError as error:
        raise ImportError("fair-esm is required to generate ESM embeddings.") from error

    if args.model_path:
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(args.model_path)
    else:
        loader = getattr(esm.pretrained, args.model_name, None)
        if loader is None:
            raise ValueError(f"Unknown esm.pretrained model: {args.model_name}")
        model, alphabet = loader()

    if args.backend == "single":
        model = model.to(device)
    else:
        if device.type != "cuda":
            raise ValueError("The FSDP backend requires CUDA.")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("This wrapper currently supports FSDP CPU offload with one torchrun process.")
        try:
            from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
            from fairscale.nn.wrap import enable_wrap, wrap
        except ImportError as error:
            raise ImportError("fairscale is required for --backend fsdp.") from error
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
        parameters = {
            "mixed_precision": True,
            "flatten_parameters": True,
            "state_dict_device": torch.device("cpu"),
            "cpu_offload": True,
        }
        with enable_wrap(wrapper_cls=FSDP, **parameters):
            for name, child in model.named_children():
                if name == "layers":
                    for layer_name, layer in child.named_children():
                        setattr(child, layer_name, wrap(layer))
            model = wrap(model)
    model.eval()
    return model, alphabet


def atomic_save(values: dict[str, torch.Tensor], path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(values, temporary_path)
    temporary_path.replace(path)


def embed_batch(model, alphabet, records: list[tuple[str, str]], device: torch.device):
    converter = alphabet.get_batch_converter()
    _, _, tokens = converter(records)
    tokens = tokens.to(device)
    lengths = (tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        results = model(tokens, repr_layers=[ESM_LAYER], return_contacts=False)
    representations = results["representations"][ESM_LAYER]
    output = {}
    for index, (identifier, _) in enumerate(records):
        sequence_length = int(lengths[index].item())
        vector = representations[index, 1 : sequence_length - 1].mean(0).detach().cpu().float()
        if vector.shape != (ESM_DIMENSION,) or not torch.isfinite(vector).all():
            raise ValueError(f"Unexpected ESM embedding for {identifier}: {tuple(vector.shape)}")
        output[identifier] = vector
    return output


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path = Path(args.failure_report).resolve() if args.failure_report else output_path.with_name("esm_embedding_failures.csv")

    records, failures = validate_records(
        read_sequences(input_path, args.id_column, args.sequence_column),
        max_length=args.max_length,
        long_sequence_policy=args.long_sequence_policy,
    )
    records.sort(key=lambda record: len(record[1]))
    existing: dict[str, torch.Tensor] = {}
    if args.resume and output_path.is_file():
        loaded = torch.load(output_path, map_location="cpu")
        existing = {
            str(key): torch.as_tensor(value).detach().cpu().float().flatten()
            for key, value in loaded.items()
            if torch.as_tensor(value).numel() == ESM_DIMENSION
        }
    records = [record for record in records if record[0] not in existing]

    device = choose_device(args.device)
    model, alphabet = load_model(args, device)
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        try:
            existing.update(embed_batch(model, alphabet, batch, device))
        except Exception as batch_error:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            for record in batch:
                try:
                    existing.update(embed_batch(model, alphabet, [record], device))
                except Exception as error:
                    failures.append(
                        {"protein_id": record[0], "stage": "embedding", "error": f"{type(error).__name__}: {error}"}
                    )
            print(f"Batch {start // args.batch_size + 1} required individual retries: {batch_error}")
        atomic_save(existing, output_path)
        print(f"Embedded {min(start + args.batch_size, len(records))}/{len(records)} pending sequences")

    with failure_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["protein_id", "stage", "error"])
        writer.writeheader()
        writer.writerows(failures)
    model_reference = args.model_path or args.model_name
    manifest = {
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "model": model_reference,
        "model_sha256": file_sha256(Path(args.model_path)) if args.model_path else None,
        "backend": args.backend,
        "pooling": "mean over residue representations excluding BOS and EOS",
        "representation_layer": ESM_LAYER,
        "embedding_dimension": ESM_DIMENSION,
        "max_length": args.max_length,
        "long_sequence_policy": args.long_sequence_policy,
        "successful_sequences": len(existing),
        "failed_sequences": len(failures),
    }
    with output_path.with_suffix(output_path.suffix + ".manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
