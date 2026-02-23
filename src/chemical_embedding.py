#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Public, single-file script (GitHub-ready)

Pipeline:
1) Read an input CSV containing compound identifiers and SMILES
2) Compute filter annotations: PAINS, SA score, QED
3) Filter compounds by fixed thresholds
4) Generate embeddings (fixed behavior):
   - Chemical Checker signatures via Signaturizer (optional; disabled by default)
   - ChemBERTa embeddings (optional; disabled by default)
   - Morgan fingerprints (enabled by default)

Only one CLI argument is supported:
  --input_csv  Path to the input CSV file

Assumptions about the input CSV:
- Must contain columns: "Name" and "SMILES" (see NAME_COL / SMILES_COL below)

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
from typing import Dict, Any, Optional, List

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
NAME_COL = "Name"
SMILES_COL = "SMILES"

# Output layout
OUTPUT_DIR = "./embeddings"
EMB_DIR = os.path.join(OUTPUT_DIR, "Chemical_embeddings")

# Filtering criteria (fixed)
EXCLUDE_PAINS = True
SA_MAX = 5.0
QED_MIN = 0.5

# Embeddings to generate (fixed)
RUN_CHEMICAL_CHECKER = True   # Set True if you want to generate Chemical Checker signatures
RUN_CHEMBERTA = True          # Set True if you want to generate ChemBERTa embeddings
RUN_MORGAN = True             # Morgan fingerprints are lightweight and enabled by default

# Chemical Checker (fixed; only used if RUN_CHEMICAL_CHECKER=True)
CC_VERSION = "current"
CC_SPACES = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5"]
# For GitHub: keep this relative (e.g., repository contains)
CC_PARAM_DIR = "path/to/chemical_checker_params" 

# ChemBERTa (fixed; only used if RUN_CHEMBERTA=True)
CHEMBERTA_TOKENIZER = "DeepChem/SmilesTokenizer_PubChem_1M"
CHEMBERTA_MODEL = "DeepChem/ChemBERTa-77M-MLM"
CHEMBERTA_BATCH_SIZE = 16
DEVICE = "auto"               # "auto", "cpu", or "cuda"

# Morgan fingerprints (fixed; only used if RUN_MORGAN=True)
MORGAN_BITS = 1024
MORGAN_RADIUS = 2

# =============================================================================
# Optional dependencies
# =============================================================================


from signaturizer import Signaturizer  # type: ignore
HAS_SIGNATURIZER = True

from transformers import AutoModel, RobertaTokenizer, pipeline  # type: ignore
HAS_TRANSFORMERS = True

# SA score (try common import locations)
import sascorer


# =============================================================================
# CLI (only input_csv)
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compound filtering + embedding generation (fixed config)")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to input CSV containing compounds (must include Name and SMILES columns).")
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

def select_device() -> torch.device:
    if DEVICE == "cpu":
        return torch.device("cpu")
    if DEVICE == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Embedding generators
# =============================================================================

def run_chemical_checker_signatures(df: pd.DataFrame) -> None:
    if not HAS_SIGNATURIZER:
        raise ImportError("signaturizer is not installed or could not be imported.")
    if not os.path.isdir(CC_PARAM_DIR):
        raise FileNotFoundError(f"CC_PARAM_DIR not found: {CC_PARAM_DIR}")

    model_paths = [os.path.join(CC_PARAM_DIR, s) for s in CC_SPACES]
    for p in model_paths:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"CC space directory not found: {p}")

    print(f"Initializing Signaturizer with spaces: {','.join(CC_SPACES)}")
    sign = Signaturizer(model_name=model_paths, local=True, version=CC_VERSION)

    out: Dict[str, Any] = {}
    print("Starting Chemical Checker signature generation...")
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Chemical Checker"):
        name = str(getattr(row, NAME_COL))
        smiles = str(getattr(row, SMILES_COL))
        try:
            res = sign.predict(smiles)
            # `res.signature` is often shape (1, n_features). Convert to a 1D torch.Tensor.
            sig = np.asarray(res.signature)
            sig = np.squeeze(sig)
            if sig.ndim != 1:
                raise ValueError(f"Unexpected signature shape for {name}: {sig.shape}")
            out[name] = torch.from_numpy(sig).float()
        except Exception as e:
            logger.warning("Failed CC signature for %s: %s", name, str(e))

    if len(out) == 0:
        logger.warning("No Chemical Checker signatures were generated; output file will be empty.")

    path = os.path.join(EMB_DIR, "chemical_checker.pt")
    torch.save(out, path)
    print(f"Saved Chemical Checker signatures to: {path}")

@torch.no_grad()
def run_chemberta_embeddings(df: pd.DataFrame) -> None:
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers is not installed or could not be imported.")

    device = select_device()
    print(f"Loading ChemBERTa tokenizer: {CHEMBERTA_TOKENIZER}")
    tokenizer = RobertaTokenizer.from_pretrained(CHEMBERTA_TOKENIZER)

    print(f"Loading ChemBERTa model: {CHEMBERTA_MODEL}")
    model = AutoModel.from_pretrained(CHEMBERTA_MODEL)
    model.eval()

    names = df[NAME_COL].astype(str).tolist()
    smiles_list = df[SMILES_COL].astype(str).tolist()

    out: Dict[str, torch.Tensor] = {}
    feature_extractor = pipeline("feature-extraction", model=model, tokenizer=tokenizer)

    for mol in df.itertuples():
        vec = feature_extractor(mol[2])
        vec = torch.tensor(vec)
        out[str(mol[1])] = vec[0][0]

    path = os.path.join(EMB_DIR, "chemberta-2.pt")
    torch.save(out, path)
    print(f"Saved ChemBERTa embeddings to: {path}")

class MorganFeaturizer:
    def __init__(self, n_bits: int = 2048, radius: int = 2):
        self.n_bits = int(n_bits)
        self.radius = int(radius)

    def smiles_to_morgan(self, smiles: str) -> np.ndarray:
        try:
            can = Chem.CanonSmiles(smiles)
            mol = Chem.MolFromSmiles(can)
            if mol is None:
                return np.zeros((self.n_bits,), dtype=np.int8)

            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
            arr = np.zeros((self.n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except Exception:
            return np.zeros((self.n_bits,), dtype=np.int8)

    def transform_to_tensor(self, smiles: str) -> torch.Tensor:
        return torch.from_numpy(self.smiles_to_morgan(smiles)).float()

def run_morgan_fingerprints(df: pd.DataFrame) -> None:
    featurizer = MorganFeaturizer(n_bits=MORGAN_BITS, radius=MORGAN_RADIUS)
    out: Dict[str, torch.Tensor] = {}

    print(f"Starting Morgan fingerprint generation (nBits={MORGAN_BITS}, radius={MORGAN_RADIUS})...")
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="MorganFP"):
        name = str(getattr(row, NAME_COL))
        smiles = str(getattr(row, SMILES_COL))
        out[name] = featurizer.transform_to_tensor(smiles)

    path = os.path.join(EMB_DIR, "morgan_fingerprint.pt")
    torch.save(out, path)
    print(f"Saved Morgan fingerprints to: {path}")

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    ensure_dir(OUTPUT_DIR)
    ensure_dir(EMB_DIR)

    df = pd.read_csv(args.input_csv)
    if NAME_COL not in df.columns or SMILES_COL not in df.columns:
        raise KeyError(
            f"Input CSV must contain columns '{NAME_COL}' and '{SMILES_COL}'. "
            f"Found columns: {list(df.columns)}"
        )

    pains_filter = build_pains_filter()

    df_with_filters = add_filter_columns(df, pains_filter)
    summarize_filtering(df_with_filters)

    df_filtered = apply_filters(df_with_filters)

    # Save filtered CSV (drop RDKit Mol objects)
    filtered_csv_path = os.path.join(OUTPUT_DIR, "data_with_filters.csv")
    df_filtered.drop(columns=["ROMol"], errors="ignore").to_csv(filtered_csv_path, index=False)
    print(f"Saved filtered CSV to: {filtered_csv_path}")

    # Use canonical SMILES for downstream computations if you want strict consistency:
    # df_filtered[SMILES_COL] = df_filtered["CanonSMILES"]

    if RUN_CHEMICAL_CHECKER:
        print("Running Chemical Checker signature generation...")
        run_chemical_checker_signatures(df_filtered)

    if RUN_CHEMBERTA:
        print("Running ChemBERTa embedding generation...")
        run_chemberta_embeddings(df_filtered)

    if RUN_MORGAN:
        print("Running Morgan fingerprint generation...")
        run_morgan_fingerprints(df_filtered)

    print("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)