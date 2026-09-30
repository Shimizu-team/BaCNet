"""Training data loading and validation for BaCNet.

``target_bct`` is treated as an already transformed training target.  This
module deliberately performs no Box-Cox transformation, scaling, clipping, or
thresholding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from model import (
    BACNET_INPUT_DIM,
    CC_EMBEDDING_DIM,
    CHEMBERTA_EMBEDDING_DIM,
    MORGAN_EMBEDDING_DIM,
    PROTEIN_EMBEDDING_DIM,
)


REQUIRED_LINK_COLUMNS = {"protein_id", "compound_id", "target_bct", "split"}
VALID_SPLITS = ("train", "validation", "test")
EMBEDDING_DIMS = {
    "protein": PROTEIN_EMBEDDING_DIM,
    "morgan": MORGAN_EMBEDDING_DIM,
    "chemical_checker": CC_EMBEDDING_DIM,
    "chemberta": CHEMBERTA_EMBEDDING_DIM,
}


@dataclass
class TrainingInputs:
    links: pd.DataFrame
    protein: Dict[str, torch.Tensor]
    morgan: Dict[str, torch.Tensor]
    chemical_checker: Dict[str, torch.Tensor]
    chemberta: Dict[str, torch.Tensor]


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


def load_embedding_dict(path: str | Path, label: str) -> Dict[str, torch.Tensor]:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"{label} embedding file not found: {input_path}")
    values = _torch_load(input_path)
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} embeddings must be a dictionary: {input_path}")
    return {str(key): torch.as_tensor(value).detach().cpu().float().flatten() for key, value in values.items()}


def load_training_inputs(
    links_path: str | Path,
    protein_path: str | Path,
    morgan_path: str | Path,
    chemical_checker_path: str | Path,
    chemberta_path: str | Path,
) -> TrainingInputs:
    links_file = Path(links_path)
    if not links_file.is_file():
        raise FileNotFoundError(f"links file not found: {links_file}")
    links = pd.read_csv(links_file)
    return TrainingInputs(
        links=links,
        protein=load_embedding_dict(protein_path, "protein"),
        morgan=load_embedding_dict(morgan_path, "Morgan"),
        chemical_checker=load_embedding_dict(chemical_checker_path, "Chemical Checker"),
        chemberta=load_embedding_dict(chemberta_path, "ChemBERTa"),
    )


def validate_training_inputs(inputs: TrainingInputs) -> pd.DataFrame:
    """Return a row-level validation report. An empty report means valid."""
    links = inputs.links
    errors: list[dict[str, object]] = []
    missing_columns = sorted(REQUIRED_LINK_COLUMNS - set(links.columns))
    if missing_columns:
        return pd.DataFrame(
            [{"row": "", "field": "links", "identifier": "", "error": f"missing columns: {missing_columns}"}]
        )

    duplicate_pair_columns = ["protein_id", "compound_id"]
    if "pair_id" in links.columns:
        duplicate_pair_columns = ["pair_id"]
    for index in links.index[links.duplicated(duplicate_pair_columns, keep=False)]:
        errors.append(
            {"row": int(index), "field": "links", "identifier": "", "error": "duplicated training pair"}
        )

    for index, row in links.iterrows():
        protein_id = str(row["protein_id"])
        compound_id = str(row["compound_id"])
        split = str(row["split"]).strip().lower()
        try:
            target = float(row["target_bct"])
            if not np.isfinite(target):
                raise ValueError("non-finite value")
        except (TypeError, ValueError) as error:
            errors.append(
                {"row": int(index), "field": "target_bct", "identifier": "", "error": str(error)}
            )
        if split not in VALID_SPLITS:
            errors.append(
                {"row": int(index), "field": "split", "identifier": split, "error": "expected train, validation, or test"}
            )

        ids = {
            "protein": (protein_id, inputs.protein),
            "morgan": (compound_id, inputs.morgan),
            "chemical_checker": (compound_id, inputs.chemical_checker),
            "chemberta": (compound_id, inputs.chemberta),
        }
        for embedding_name, (identifier, mapping) in ids.items():
            if identifier not in mapping:
                errors.append(
                    {"row": int(index), "field": embedding_name, "identifier": identifier, "error": "embedding not found"}
                )

    # Validate every referenced vector once, rather than for every pair.
    referenced = {
        "protein": (set(links["protein_id"].astype(str)), inputs.protein),
        "morgan": (set(links["compound_id"].astype(str)), inputs.morgan),
        "chemical_checker": (set(links["compound_id"].astype(str)), inputs.chemical_checker),
        "chemberta": (set(links["compound_id"].astype(str)), inputs.chemberta),
    }
    for embedding_name, (identifiers, mapping) in referenced.items():
        for identifier in sorted(identifiers & set(mapping)):
            vector = mapping[identifier]
            expected_dim = EMBEDDING_DIMS[embedding_name]
            if vector.numel() != expected_dim:
                errors.append(
                    {
                        "row": "",
                        "field": embedding_name,
                        "identifier": identifier,
                        "error": f"expected {expected_dim} dimensions, got {vector.numel()}",
                    }
                )
            elif not torch.isfinite(vector).all():
                errors.append(
                    {"row": "", "field": embedding_name, "identifier": identifier, "error": "non-finite embedding value"}
                )

    for split in VALID_SPLITS:
        if not (links["split"].astype(str).str.lower() == split).any():
            errors.append(
                {"row": "", "field": "split", "identifier": split, "error": "split is empty"}
            )
    return pd.DataFrame(errors, columns=["row", "field", "identifier", "error"])


class BaCNetTrainingDataset(Dataset):
    def __init__(self, links: pd.DataFrame, inputs: TrainingInputs):
        links = links.copy()
        if "_source_index" not in links:
            links["_source_index"] = links.index
        self.links = links.reset_index(drop=True)
        self.inputs = inputs

    def __len__(self) -> int:
        return len(self.links)

    def __getitem__(self, index: int):
        row = self.links.iloc[index]
        protein_id = str(row["protein_id"])
        compound_id = str(row["compound_id"])
        features = torch.cat(
            [
                self.inputs.protein[protein_id],
                self.inputs.morgan[compound_id],
                self.inputs.chemical_checker[compound_id],
                self.inputs.chemberta[compound_id],
            ]
        ).float()
        if features.numel() != BACNET_INPUT_DIM:
            raise ValueError(
                f"Invalid BaCNet input for {protein_id}/{compound_id}: "
                f"expected {BACNET_INPUT_DIM}, got {features.numel()}"
            )
        return features, torch.tensor(float(row["target_bct"]), dtype=torch.float32), int(row["_source_index"])


def create_dataloaders(
    inputs: TrainingInputs,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> dict[str, DataLoader]:
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 because BaCNet uses BatchNorm.")
    generator = torch.Generator().manual_seed(seed)
    loaders: dict[str, DataLoader] = {}
    normalized_split = inputs.links["split"].astype(str).str.lower()
    for split in VALID_SPLITS:
        subset = inputs.links.loc[normalized_split == split].copy()
        loaders[split] = DataLoader(
            BaCNetTrainingDataset(subset, inputs),
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=split == "train" and len(subset) % batch_size == 1,
            generator=generator if split == "train" else None,
        )
    return loaders


def make_synthetic_inputs(seed: int = 123) -> TrainingInputs:
    """Build small, deterministic data for tests only; never for scientific use."""
    generator = torch.Generator().manual_seed(seed)
    protein_ids = [f"synthetic_protein_{index}" for index in range(6)]
    compound_ids = [f"synthetic_compound_{index}" for index in range(8)]
    split_sizes = {"train": 16, "validation": 6, "test": 6}
    rows = []
    offset = 0
    for split, size in split_sizes.items():
        for local_index in range(size):
            index = offset + local_index
            rows.append(
                {
                    "pair_id": f"synthetic_pair_{index}",
                    "protein_id": protein_ids[index % len(protein_ids)],
                    "compound_id": compound_ids[(index // len(protein_ids)) % len(compound_ids)],
                    "target_bct": float((index % 11) / 10),
                    "split": split,
                }
            )
        offset += size

    def vectors(ids: Sequence[str], dimension: int) -> Dict[str, torch.Tensor]:
        return {identifier: torch.randn(dimension, generator=generator) for identifier in ids}

    return TrainingInputs(
        links=pd.DataFrame(rows),
        protein=vectors(protein_ids, PROTEIN_EMBEDDING_DIM),
        morgan=vectors(compound_ids, MORGAN_EMBEDDING_DIM),
        chemical_checker=vectors(compound_ids, CC_EMBEDDING_DIM),
        chemberta=vectors(compound_ids, CHEMBERTA_EMBEDDING_DIM),
    )
