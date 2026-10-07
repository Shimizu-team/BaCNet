#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Public, single-file script (GitHub-ready)

Pipeline:
1) Read an input CSV containing compound identifiers and SMILES
2) Compute filter annotations: PAINS, SA score, QED
3) Filter compounds by fixed thresholds
4) Generate Chemical Checker, ChemBERTa, and Morgan fingerprint embeddings
5) Write per-compound embedding failures to embedding_failures.csv

CLI arguments:
  --input_csv      Path to the input CSV file
  --cc-param-dir   Directory containing A1-A5 and B1-B5 Chemical Checker models
  --output-dir     Directory in which generated files are written
  --admet_ai_csv   Optional ADMET-AI result CSV file
  --admet_filter   Optional ADMET-AI filter level

Assumptions about the input CSV:
- Must contain columns: "Compound_ID" and "SMILES" (see NAME_COL / SMILES_COL below)

Notes:
- SA score uses the common RDKit Contrib implementation (SA_Score/sascorer.py) if available.
  If unavailable, SA filtering is skipped while the script remains runnable.
- Chemical Checker signatures require `signaturizer` and a local CC_param directory.
- ChemBERTa requires `transformers` and will use GPU if available.
"""

from __future__ import annotations

import os
import sys
import argparse
import warnings
import logging
import hashlib
import json
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Fixed configuration (edit here if needed)
# =============================================================================

# Input schema
NAME_COL = "Compound_ID"
SMILES_COL = "SMILES"

# Filtering criteria (fixed)
EXCLUDE_PAINS = True
SA_MAX = 5.0
QED_MIN = 0.5

# ADMET-AI filtering (optional)
ADMET_AI_METRICS = {
    "AMES": {
        "flag_threshold": 0.70,
        "alert_threshold": 0.90,
    },
    "hERG": {
        "flag_threshold": 0.70,
        "alert_threshold": 0.90,
    },
    "DILI": {
        "flag_threshold": 0.70,
        "alert_threshold": 0.90,
    },
    "CYP3A4_Veith": {
        "flag_threshold": 0.70,
        "alert_threshold": 0.90,
    },
}

# Chemical Checker: BaCNet uses these ten 128-dimensional spaces (1,280 total).
CC_SPACES = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5"]
CC_EMBEDDING_DIM = 1280

# ChemBERTa (fixed; only used if RUN_CHEMBERTA=True)
CHEMBERTA_TOKENIZER = "DeepChem/SmilesTokenizer_PubChem_1M"
CHEMBERTA_MODEL = "DeepChem/ChemBERTa-77M-MLM"
CHEMBERTA_BATCH_SIZE = 16
CHEMBERTA_EMBEDDING_DIM = 384
DEVICE = "auto"               # "auto", "cpu", or "cuda"

# Morgan fingerprints (fixed; only used if RUN_MORGAN=True)
MORGAN_BITS = 1024
MORGAN_RADIUS = 2

# =============================================================================
# Optional dependencies
# =============================================================================


try:
    from signaturizer import Signaturizer  # type: ignore
    HAS_SIGNATURIZER = True
except ImportError:
    Signaturizer = None  # type: ignore
    HAS_SIGNATURIZER = False

try:
    from transformers import AutoModel, RobertaTokenizer  # type: ignore
    HAS_TRANSFORMERS = True
except ImportError:
    AutoModel = None  # type: ignore
    RobertaTokenizer = None  # type: ignore
    HAS_TRANSFORMERS = False

# SA score (try common import locations)
import sascorer


# =============================================================================
# CLI (only input_csv)
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compound filtering + embedding generation (fixed config)")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to input CSV containing Compound_ID and SMILES columns.")
    parser.add_argument(
        "--cc-param-dir",
        type=str,
        required=False,
        help="Directory containing the A1-A5 and B1-B5 Chemical Checker model directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="embeddings",
        help="Directory in which filtered data and embedding files are written (default: embeddings).",
    )
    parser.add_argument(
        "--admet_ai_csv",
        type=str,
        default=None,
        help="Optional ADMET-AI output CSV path. If provided, ADMET-AI annotations/filtering are applied.",
    )
    parser.add_argument(
        "--admet_filter",
        choices=["none", "flag", "alert"],
        default="none",
        help="ADMET-AI filtering level. 'alert' removes alert compounds; 'flag' removes flag/alert compounds.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["chemical_checker", "chemberta", "morgan"],
        default=["chemical_checker", "chemberta", "morgan"],
        help="Embedding methods to run (default: all three).",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEVICE)
    parser.add_argument("--chemberta-batch-size", type=int, default=CHEMBERTA_BATCH_SIZE)
    parser.add_argument("--resume", action="store_true", help="Keep valid embeddings already present in output files.")
    return parser.parse_args()

# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def canonicalize_smiles(smiles: str) -> Optional[str]:
    try:
        return Chem.CanonSmiles(smiles)
    except Exception:
        return None

def build_pains_filter() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)

def compute_sa_score(mol: Chem.Mol) -> Optional[float]:
    return float(sascorer.calculateScore(mol)) 

def add_filter_columns(df: pd.DataFrame, pains_filter: FilterCatalog) -> pd.DataFrame:
    """
    Add columns:
      - CanonSMILES
      - ROMol
      - PAINS (bool)
      - SA_score (float or NaN)
      - QED (float)
    """
    df = df.copy()

    tqdm.pandas(desc="Canonicalizing SMILES")
    df["CanonSMILES"] = df[SMILES_COL].progress_apply(
        lambda s: canonicalize_smiles(str(s)) if pd.notna(s) else None
    )

    tqdm.pandas(desc="SMILES -> RDKit Mol")
    df["ROMol"] = df["CanonSMILES"].progress_apply(
        lambda s: Chem.MolFromSmiles(s) if isinstance(s, str) else None
    )

    # Drop invalid molecules early for robustness
    n_before = len(df)
    df = df[df["ROMol"].notnull()].copy()
    n_after = len(df)
    print(f"Dropped {n_before - n_after} invalid molecules (MolFromSmiles failed).")

    tqdm.pandas(desc="Computing PAINS matches")
    df["PAINS"] = df["ROMol"].progress_apply(lambda m: bool(pains_filter.HasMatch(m)))

    tqdm.pandas(desc="Computing SA score")
    df["SA_score"] = df["ROMol"].progress_apply(lambda m: compute_sa_score(m))
    df["SA_score"] = pd.to_numeric(df["SA_score"], errors="coerce")

    tqdm.pandas(desc="Computing QED")
    df["QED"] = df["ROMol"].progress_apply(lambda m: float(QED.qed(m)))

    return df

def summarize_filtering(df: pd.DataFrame) -> None:
    total = len(df)
    pains_n = int((df["PAINS"] == False).sum())
    sa_n = int((df["SA_score"] < SA_MAX).sum())
    qed_n = int((df["QED"] > QED_MIN).sum())

    print(f"#data before filtering: {total}")
    print(f"PAINS matches: {pains_n}/{total}")

    print(f"SA_score < {SA_MAX:.2f}: {sa_n}/{total}")
    print(f"QED > {QED_MIN:.2f}: {qed_n}/{total}")

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    df_f = df.copy()

    df_f = df_f[df_f["PAINS"] == False]  
    df_f = df_f[df_f["SA_score"] <= SA_MAX]
    df_f = df_f[df_f["QED"] >= QED_MIN]

    print(f"#data after filtering: {len(df_f)}" )
    return df_f

def add_filter_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Record the fixed PAINS/SA/QED decision for every valid molecule."""
    result = df.copy()

    def reasons(row: pd.Series) -> str:
        failures = []
        if bool(row["PAINS"]):
            failures.append("PAINS")
        if pd.isna(row["SA_score"]):
            failures.append("invalid_SA_score")
        elif float(row["SA_score"]) > SA_MAX:
            failures.append(f"SA_score>{SA_MAX:g}")
        if pd.isna(row["QED"]):
            failures.append("invalid_QED")
        elif float(row["QED"]) < QED_MIN:
            failures.append(f"QED<{QED_MIN:g}")
        return ";".join(failures)

    result["Filter_failure_reason"] = result.apply(reasons, axis=1)
    result["Filter_pass"] = result["Filter_failure_reason"].eq("")
    return result

def judge_admet_metric(value: Any, flag_threshold: float, alert_threshold: float) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "missing"

    if pd.isna(numeric_value):
        return "missing"

    if numeric_value >= alert_threshold:
        return "alert"
    if numeric_value >= flag_threshold:
        return "flag"
    return "keep"

def load_admet_ai_data(path: str) -> pd.DataFrame:
    admet_ai_data = pd.read_csv(path)
    required_columns = ["SMILES", *ADMET_AI_METRICS.keys()]
    missing_columns = [column for column in required_columns if column not in admet_ai_data.columns]
    if missing_columns:
        raise ValueError(f"{missing_columns} is not found in {path}")

    admet_ai_data = admet_ai_data.copy()
    admet_ai_data["SMILES"] = admet_ai_data["SMILES"].astype(str).str.strip()

    duplicated_smiles = int(admet_ai_data["SMILES"].duplicated().sum())
    if duplicated_smiles:
        logger.warning("Duplicated ADMET-AI SMILES were found: %d. Keeping the first record.", duplicated_smiles)
        admet_ai_data = admet_ai_data.drop_duplicates(subset="SMILES", keep="first")

    return admet_ai_data

def add_admet_ai_judgements(data: pd.DataFrame) -> pd.DataFrame:
    result_data = data.copy()

    for metric_name, metric_config in ADMET_AI_METRICS.items():
        judgement_column = f"{metric_name}_judgement"
        result_data[judgement_column] = result_data[metric_name].apply(
            lambda value: judge_admet_metric(
                value,
                metric_config["flag_threshold"],
                metric_config["alert_threshold"],
            )
        )

    result_data["admet_ai_flag_metrics"] = result_data.apply(
        lambda row: ",".join(
            metric_name
            for metric_name in ADMET_AI_METRICS
            if row[f"{metric_name}_judgement"] in {"flag", "alert"}
        ),
        axis=1,
    )
    result_data["admet_ai_alert_metrics"] = result_data.apply(
        lambda row: ",".join(
            metric_name
            for metric_name in ADMET_AI_METRICS
            if row[f"{metric_name}_judgement"] == "alert"
        ),
        axis=1,
    )
    result_data["admet_ai_has_flag"] = result_data["admet_ai_flag_metrics"].astype(bool)
    result_data["admet_ai_has_alert"] = result_data["admet_ai_alert_metrics"].astype(bool)

    return result_data

def apply_admet_filter(data: pd.DataFrame, admet_filter: str) -> pd.DataFrame:
    if admet_filter == "alert":
        return data[~data["admet_ai_has_alert"]].copy()

    if admet_filter == "flag":
        return data[~data["admet_ai_has_flag"]].copy()

    return data

def merge_admet_ai_results(df: pd.DataFrame, admet_ai_csv: str, admet_filter: str) -> pd.DataFrame:
    result_data = df.copy()
    result_data[SMILES_COL] = result_data[SMILES_COL].astype(str).str.strip()

    admet_ai_data = load_admet_ai_data(admet_ai_csv)
    result_data = result_data.merge(admet_ai_data, on=SMILES_COL, how="inner")

    skipped_count = len(df) - len(result_data)
    if skipped_count:
        print(f"Skip: ADMET-AI result was not available for {skipped_count} compounds")

    result_data = add_admet_ai_judgements(result_data)
    result_data = apply_admet_filter(result_data, admet_filter)

    print(f"#data after ADMET-AI filtering: {len(result_data)}")
    return result_data

def select_device(device_setting: str = DEVICE) -> torch.device:
    if device_setting == "cpu":
        return torch.device("cpu")
    if device_setting == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Embedding generators
# =============================================================================

def failure_record(embedding_type: str, name: str, smiles: str, error: Exception) -> Dict[str, str]:
    return {
        "Embedding_type": embedding_type,
        "Compound_ID": name,
        "SMILES": smiles,
        "Error_type": type(error).__name__,
        "Error_message": str(error),
    }


def load_existing_embeddings(path: str, expected_dim: int, resume: bool) -> Dict[str, torch.Tensor]:
    if not resume or not os.path.isfile(path):
        return {}
    values = torch.load(path, map_location="cpu")
    if not isinstance(values, dict):
        raise TypeError(f"Existing embedding file must contain a dictionary: {path}")
    output: Dict[str, torch.Tensor] = {}
    for key, value in values.items():
        vector = torch.as_tensor(value).detach().cpu().float().flatten()
        if vector.numel() == expected_dim and torch.isfinite(vector).all():
            output[str(key)] = vector
    return output


def run_chemical_checker_signatures(
    df: pd.DataFrame,
    cc_param_dir: str,
    emb_dir: str,
    resume: bool = False,
) -> list[Dict[str, str]]:
    if not HAS_SIGNATURIZER:
        raise ImportError("signaturizer is not installed or could not be imported.")
    if not os.path.isdir(cc_param_dir):
        raise FileNotFoundError(f"Chemical Checker parameter directory not found: {cc_param_dir}")

    model_paths = [os.path.join(cc_param_dir, s) for s in CC_SPACES]
    for p in model_paths:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"CC space directory not found: {p}")

    print(f"Initializing Signaturizer with spaces: {','.join(CC_SPACES)}")
    sign = Signaturizer(model_name=model_paths, local=True)

    path = os.path.join(emb_dir, "chemical_checker.pt")
    out = load_existing_embeddings(path, CC_EMBEDDING_DIM, resume)
    failures: list[Dict[str, str]] = []
    print("Starting Chemical Checker signature generation...")
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Chemical Checker"):
        name = str(getattr(row, NAME_COL))
        smiles = str(getattr(row, SMILES_COL))
        if name in out:
            continue
        try:
            res = sign.predict(smiles)
            # `res.signature` is often shape (1, n_features). Convert to a 1D torch.Tensor.
            sig = np.asarray(res.signature)
            sig = np.squeeze(sig)
            if sig.shape != (CC_EMBEDDING_DIM,):
                raise ValueError(
                    f"Unexpected Chemical Checker shape for {name}: {sig.shape}; "
                    f"expected ({CC_EMBEDDING_DIM},) from {len(CC_SPACES)} spaces"
                )
            out[name] = torch.from_numpy(sig).float()
        except Exception as e:
            logger.warning("Failed CC signature for %s: %s", name, str(e))
            failures.append(failure_record("chemical_checker", name, smiles, e))

    if len(out) == 0:
        logger.warning("No Chemical Checker signatures were generated; output file will be empty.")

    torch.save(out, path)
    print(f"Saved Chemical Checker signatures to: {path}")
    return failures

@torch.no_grad()
def run_chemberta_embeddings(
    df: pd.DataFrame,
    emb_dir: str,
    device_setting: str = DEVICE,
    batch_size: int = CHEMBERTA_BATCH_SIZE,
    resume: bool = False,
) -> list[Dict[str, str]]:
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers is not installed or could not be imported.")

    device = select_device(device_setting)
    print(f"Loading ChemBERTa tokenizer: {CHEMBERTA_TOKENIZER}")
    tokenizer = RobertaTokenizer.from_pretrained(CHEMBERTA_TOKENIZER)

    print(f"Loading ChemBERTa model: {CHEMBERTA_MODEL}")
    model = AutoModel.from_pretrained(CHEMBERTA_MODEL).to(device)
    model.eval()

    path = os.path.join(emb_dir, "chemberta-2.pt")
    out = load_existing_embeddings(path, CHEMBERTA_EMBEDDING_DIM, resume)
    pending = df[~df[NAME_COL].astype(str).isin(out)].copy()
    names = pending[NAME_COL].astype(str).tolist()
    smiles_list = pending[SMILES_COL].astype(str).tolist()
    failures: list[Dict[str, str]] = []

    def embed_batch(batch_names: list[str], batch_smiles: list[str]) -> None:
        encoded = tokenizer(
            batch_smiles,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        cls_embeddings = model(**encoded).last_hidden_state[:, 0, :].detach().cpu().float()
        if cls_embeddings.ndim != 2 or cls_embeddings.shape[1] != CHEMBERTA_EMBEDDING_DIM:
            raise ValueError(
                f"Unexpected ChemBERTa output shape: {tuple(cls_embeddings.shape)}; "
                f"expected (batch_size, {CHEMBERTA_EMBEDDING_DIM})"
            )
        for name, embedding in zip(batch_names, cls_embeddings):
            out[name] = embedding

    for start in tqdm(range(0, len(names), batch_size), desc="ChemBERTa batches"):
        batch_names = names[start:start + batch_size]
        batch_smiles = smiles_list[start:start + batch_size]
        try:
            embed_batch(batch_names, batch_smiles)
        except Exception as batch_error:
            logger.warning(
                "ChemBERTa batch starting at row %d failed (%s); retrying compounds individually.",
                start,
                str(batch_error),
            )
            for name, smiles in zip(batch_names, batch_smiles):
                try:
                    embed_batch([name], [smiles])
                except Exception as error:
                    logger.warning("Failed ChemBERTa embedding for %s: %s", name, str(error))
                    failures.append(failure_record("chemberta", name, smiles, error))

    torch.save(out, path)
    print(f"Saved ChemBERTa embeddings to: {path}")
    return failures

class MorganFeaturizer:
    def __init__(self, n_bits: int = 2048, radius: int = 2):
        self.n_bits = int(n_bits)
        self.radius = int(radius)

    def smiles_to_morgan(self, smiles: str) -> np.ndarray:
        can = Chem.CanonSmiles(smiles)
        mol = Chem.MolFromSmiles(can)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles}")

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        arr = np.zeros((self.n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def transform_to_tensor(self, smiles: str) -> torch.Tensor:
        return torch.from_numpy(self.smiles_to_morgan(smiles)).float()

def run_morgan_fingerprints(df: pd.DataFrame, emb_dir: str, resume: bool = False) -> list[Dict[str, str]]:
    featurizer = MorganFeaturizer(n_bits=MORGAN_BITS, radius=MORGAN_RADIUS)
    path = os.path.join(emb_dir, "morgan_fingerprint.pt")
    out = load_existing_embeddings(path, MORGAN_BITS, resume)
    failures: list[Dict[str, str]] = []

    print(f"Starting Morgan fingerprint generation (nBits={MORGAN_BITS}, radius={MORGAN_RADIUS})...")
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="MorganFP"):
        name = str(getattr(row, NAME_COL))
        smiles = str(getattr(row, SMILES_COL))
        if name in out:
            continue
        try:
            embedding = featurizer.transform_to_tensor(smiles)
            if embedding.shape != (MORGAN_BITS,):
                raise ValueError(
                    f"Unexpected Morgan fingerprint shape for {name}: {tuple(embedding.shape)}; "
                    f"expected ({MORGAN_BITS},)"
                )
            out[name] = embedding
        except Exception as error:
            logger.warning("Failed Morgan fingerprint for %s: %s", name, str(error))
            failures.append(failure_record("morgan", name, smiles, error))

    torch.save(out, path)
    print(f"Saved Morgan fingerprints to: {path}")
    return failures

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    output_dir = os.path.abspath(args.output_dir)
    emb_dir = os.path.join(output_dir, "Chemical_embeddings")
    ensure_dir(output_dir)
    ensure_dir(emb_dir)

    df = pd.read_csv(args.input_csv)
    if NAME_COL not in df.columns or SMILES_COL not in df.columns:
        raise KeyError(
            f"Input CSV must contain columns '{NAME_COL}' and '{SMILES_COL}'. "
            f"Found columns: {list(df.columns)}"
        )
    if df[NAME_COL].isna().any() or (df[NAME_COL].astype(str).str.strip() == "").any():
        raise ValueError("Compound_ID must not be empty.")
    if df[NAME_COL].astype(str).duplicated().any():
        duplicates = df.loc[df[NAME_COL].astype(str).duplicated(keep=False), NAME_COL].astype(str).unique()
        raise ValueError(f"Compound_ID values must be unique. Duplicates: {duplicates[:10].tolist()}")
    if "chemical_checker" in args.methods and not args.cc_param_dir:
        raise ValueError("--cc-param-dir is required when chemical_checker is selected.")
    if args.chemberta_batch_size < 1:
        raise ValueError("--chemberta-batch-size must be at least 1.")

    pains_filter = build_pains_filter()

    df_with_filters = add_filter_columns(df, pains_filter)
    summarize_filtering(df_with_filters)
    df_with_filters = add_filter_outcomes(df_with_filters)

    annotation_path = os.path.join(output_dir, "filter_annotations.csv")
    df_with_filters.drop(columns=["ROMol"], errors="ignore").to_csv(annotation_path, index=False)
    print(f"Saved filter annotations to: {annotation_path}")

    df_filtered = apply_filters(df_with_filters)

    if args.admet_ai_csv:
        print("Applying ADMET-AI annotations and filtering...")
        df_filtered = merge_admet_ai_results(df_filtered, args.admet_ai_csv, args.admet_filter)

    # Every embedding method consumes exactly the same canonical SMILES.
    df_filtered[SMILES_COL] = df_filtered["CanonSMILES"]

    # Save filtered CSV (drop RDKit Mol objects)
    filtered_csv_path = os.path.join(output_dir, "data_with_filters.csv")
    df_filtered.drop(columns=["ROMol"], errors="ignore").to_csv(filtered_csv_path, index=False)
    print(f"Saved filtered CSV to: {filtered_csv_path}")

    failures: list[Dict[str, str]] = []

    if "chemical_checker" in args.methods:
        print("Running Chemical Checker signature generation...")
        failures.extend(run_chemical_checker_signatures(df_filtered, args.cc_param_dir, emb_dir, args.resume))

    if "chemberta" in args.methods:
        print("Running ChemBERTa embedding generation...")
        failures.extend(
            run_chemberta_embeddings(
                df_filtered,
                emb_dir,
                device_setting=args.device,
                batch_size=args.chemberta_batch_size,
                resume=args.resume,
            )
        )

    if "morgan" in args.methods:
        print("Running Morgan fingerprint generation...")
        failures.extend(run_morgan_fingerprints(df_filtered, emb_dir, args.resume))

    failure_columns = ["Embedding_type", "Compound_ID", "SMILES", "Error_type", "Error_message"]
    failure_path = os.path.join(output_dir, "embedding_failures.csv")
    pd.DataFrame(failures, columns=failure_columns).to_csv(failure_path, index=False)
    print(f"Saved embedding failure report ({len(failures)} failures) to: {failure_path}")

    def digest(path: str) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha.update(chunk)
        return sha.hexdigest()

    output_files = {
        "chemical_checker": "chemical_checker.pt",
        "chemberta": "chemberta-2.pt",
        "morgan": "morgan_fingerprint.pt",
    }
    manifest: Dict[str, Any] = {
        "input_csv": os.path.abspath(args.input_csv),
        "input_sha256": digest(args.input_csv),
        "compound_id_column": NAME_COL,
        "smiles_source": "CanonSMILES",
        "methods": args.methods,
        "dimensions": {"chemical_checker": 1280, "chemberta": 384, "morgan": 1024},
        "models": {
            "chemical_checker_spaces": CC_SPACES,
            "chemberta_tokenizer": CHEMBERTA_TOKENIZER,
            "chemberta_model": CHEMBERTA_MODEL,
            "morgan_radius": MORGAN_RADIUS,
        },
        "outputs": {},
        "failures": len(failures),
    }
    for method in args.methods:
        output_path = os.path.join(emb_dir, output_files[method])
        manifest["outputs"][method] = {"path": os.path.abspath(output_path), "sha256": digest(output_path)}
    with open(os.path.join(output_dir, "embedding_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
