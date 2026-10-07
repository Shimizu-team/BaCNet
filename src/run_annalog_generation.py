#!/usr/bin/env python3
"""Generate one ANNalog library with an explicit random seed.

This adapter keeps the third-party ANNalog package in its own Conda environment
while providing a stable CLI for the BaCNet workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from annalog.model_handler import SMILESModelHandler
from annalog.SMILES_generator import SMILESGenerator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fix_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one seeded ANNalog compound library.")
    parser.add_argument("--input-smiles", required=True)
    parser.add_argument("--generation-class", choices=["medium", "far"], required=True)
    parser.add_argument("--generation-method", default="sampling")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--generation-number", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.generation_number < 1:
        raise ValueError("--generation-number must be at least 1")
    for path in (args.vocab_path, args.checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    fix_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    handler = SMILESModelHandler(
        src_vocab_path=str(args.vocab_path),
        trg_vocab_path=str(args.vocab_path),
        model_path=str(args.checkpoint_path),
        device=device,
    )
    generator = SMILESGenerator(handler)
    generated = generator.generate_smiles(
        input_smiles=args.input_smiles.strip(),
        generation_number=args.generation_number,
        temperature=args.temperature,
        generation_method=args.generation_method,
        prefix=0,
        filter_invalid=True,
    )

    rows = []
    for smiles, score in generated:
        if torch.is_tensor(score):
            score = score.detach().cpu().item()
        rows.append({"SMILES": str(smiles), "Score": float(score)})

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["SMILES", "Score"]).to_csv(args.output_csv, index=False)
    manifest_path = args.manifest or args.output_csv.with_suffix(".manifest.json")
    manifest = {
        "input_smiles": args.input_smiles.strip(),
        "generation_class": args.generation_class,
        "generation_method": args.generation_method,
        "temperature": args.temperature,
        "generation_number_requested": args.generation_number,
        "generation_number_returned": len(rows),
        "seed": args.seed,
        "device": device,
        "vocab_path": str(args.vocab_path.resolve()),
        "vocab_sha256": sha256(args.vocab_path),
        "checkpoint_path": str(args.checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint_path),
        "output_csv": str(args.output_csv.resolve()),
        "output_sha256": sha256(args.output_csv),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(rows)} {args.generation_class} compounds: {args.output_csv}")


if __name__ == "__main__":
    main()
