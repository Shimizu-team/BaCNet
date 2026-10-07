"""Training data loading and validation for BaCNet.

``transformed_score`` is treated as an already transformed training target. This
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


REQUIRED_LINK_COLUMNS = {"protein_id", "compound_id", "transformed_score", "split"}
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


def protein_group_column(links: pd.DataFrame) -> str:
    """Return the column used to prevent protein leakage between splits.

    ``protein_group_id`` should identify sequence-equivalent proteins when the
    same amino-acid sequence can have more than one ``protein_id``. If that
    optional column is absent, ``protein_id`` itself is used as the group.
    """
    return "protein_group_id" if "protein_group_id" in links.columns else "protein_id"


def reference_leave_one_protein_out_split(
    links: pd.DataFrame,
    protein_column: str | None = None,
    test_size: float = 0.10,
    validation_size: float = 0.10,
    seed: int = 123,
) -> pd.DataFrame:
    """Reference implementation of the dataset split used for BaCNet.

    This function is provided for methodological transparency and is not called
    by the training CLI. Training expects a preassigned ``split`` column.

    Unique protein groups are split first, so no group can occur in more than
    one subset. Ten percent of groups are assigned to test, followed by ten
    percent of the remaining groups to validation. Thus, the approximate group
    ratio is 81% train, 9% validation, and 10% test. Both steps use
    ``random_state=123`` by default, matching the original implementation.

    Use ``protein_group_id`` for ``protein_column`` when multiple identifiers
    represent the same sequence; otherwise ``protein_id`` is used.

    Protein groups retain their first-appearance order before scikit-learn
    performs the seeded shuffle, matching the original ``Series.unique()``
    procedure. Recreating an identical split therefore also requires identical
    input row order and a compatible scikit-learn implementation.
    """
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as error:
        raise ImportError("scikit-learn is required only when recreating the reference split.") from error

    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must each be between 0 and 1.")
    group_column = protein_column or protein_group_column(links)
    if group_column not in links.columns:
        raise KeyError(f"Protein grouping column not found: {group_column}")
    if links[group_column].isna().any():
        raise ValueError(f"Protein grouping column contains missing values: {group_column}")

    unique_groups = links[group_column].astype(str).unique()
    remaining_groups, test_groups = train_test_split(
        unique_groups,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    train_groups, validation_groups = train_test_split(
        remaining_groups,
        test_size=validation_size,
        random_state=seed,
        shuffle=True,
    )
    group_to_split = {str(group): "train" for group in train_groups}
    group_to_split.update({str(group): "validation" for group in validation_groups})
    group_to_split.update({str(group): "test" for group in test_groups})

    result = links.copy()
    result["split"] = result[group_column].astype(str).map(group_to_split)
    if result["split"].isna().any():
        raise RuntimeError("Some protein groups were not assigned to a split.")
    return result


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


def load_training_inputs_from_splits(
    train_links_path: str | Path,
    validation_links_path: str | Path,
    test_links_path: str | Path,
    protein_path: str | Path,
    morgan_path: str | Path,
    chemical_checker_path: str | Path,
    chemberta_path: str | Path,
    column_names: Mapping[str, str] | None = None,
) -> TrainingInputs:
    """Load authoritative train/validation/test files without resampling.

    ``column_names`` maps the canonical names ``protein_id``, ``compound_id``,
    and ``transformed_score`` to their names in the distributed CSV files. The split
    label is derived from the file itself and cannot be overridden by a CSV
    column.
    """

    source_columns = {
        "protein_id": "protein_id",
        "compound_id": "compound_id",
        "transformed_score": "transformed_score",
    }
    if column_names:
        source_columns.update({key: str(value) for key, value in column_names.items()})
    unknown = set(source_columns) - {"protein_id", "compound_id", "transformed_score"}
    if unknown:
        raise ValueError(f"Unknown canonical column name(s): {sorted(unknown)}")

    frames = []
    paths = {
        "train": Path(train_links_path),
        "validation": Path(validation_links_path),
        "test": Path(test_links_path),
    }
    rename_columns = {source: canonical for canonical, source in source_columns.items()}
    for split, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{split} links file not found: {path}")
        frame = pd.read_csv(path)
        missing = sorted(set(rename_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required column(s): {missing}")
        frame = frame.rename(columns=rename_columns)
        frame["split"] = split
        frames.append(frame)

    return TrainingInputs(
        links=pd.concat(frames, ignore_index=True),
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
            target = float(row["transformed_score"])
            if not np.isfinite(target):
                raise ValueError("non-finite value")
        except (TypeError, ValueError) as error:
            errors.append(
                {"row": int(index), "field": "transformed_score", "identifier": "", "error": str(error)}
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

    # The provided split must preserve the leave-one-protein-out boundary.
    group_column = protein_group_column(links)
    missing_group = links[group_column].isna() | (links[group_column].astype(str).str.strip() == "")
    for index in links.index[missing_group]:
        errors.append(
            {
                "row": int(index),
                "field": group_column,
                "identifier": "",
                "error": "protein group must not be empty",
            }
        )
    normalized_split = links["split"].astype(str).str.lower()
    groups_by_split = {
        split: set(links.loc[normalized_split == split, group_column].astype(str))
        for split in VALID_SPLITS
    }
    for left_index, left_split in enumerate(VALID_SPLITS):
        for right_split in VALID_SPLITS[left_index + 1 :]:
            overlap = groups_by_split[left_split] & groups_by_split[right_split]
            for group in sorted(overlap):
                errors.append(
                    {
                        "row": "",
                        "field": group_column,
                        "identifier": group,
                        "error": f"leave-one-protein-out leakage between {left_split} and {right_split}",
                    }
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
        return features, torch.tensor(float(row["transformed_score"]), dtype=torch.float32), int(row["_source_index"])


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
    protein_ids = [f"synthetic_protein_{index}" for index in range(14)]
    compound_ids = [f"synthetic_compound_{index}" for index in range(8)]
    split_sizes = {"train": 16, "validation": 6, "test": 6}
    split_proteins = {
        "train": protein_ids[:8],
        "validation": protein_ids[8:11],
        "test": protein_ids[11:14],
    }
    rows = []
    offset = 0
    for split, size in split_sizes.items():
        for local_index in range(size):
            index = offset + local_index
            rows.append(
                {
                    "pair_id": f"synthetic_pair_{index}",
                    "protein_id": split_proteins[split][local_index % len(split_proteins[split])],
                    "compound_id": compound_ids[(offset // 6 + local_index // len(split_proteins[split])) % len(compound_ids)],
                    "transformed_score": float((index % 11) / 10),
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
