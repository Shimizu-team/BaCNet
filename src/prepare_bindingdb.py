#!/usr/bin/env python3
"""Build the Pseudomonas aeruginosa quantitative-affinity datasets.

The input is ``BindingDB_All_202609_tsv.zip`` (or its extracted TSV). Ki, Kd,
and IC50 are curated independently. Ligands are canonicalized with RDKit only;
no desalting, neutralization, or tautomer standardization is performed.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator

import pandas as pd
from rdkit import Chem


SPECIES_NAME = "Pseudomonas aeruginosa"
ENDPOINT_COLUMNS = {"Ki": "Ki (nM)", "Kd": "Kd (nM)", "IC50": "IC50 (nM)"}
ORGANISM_COLUMN = "Target Source Organism According to Curator or DataSource"
CHAIN_COUNT_COLUMN = "Number of Protein Chains in Target (>1 implies a multichain complex)"
SEQUENCE_COLUMN = "BindingDB Target Chain Sequence 1"
SWISSPROT_COLUMN = "UniProt (SwissProt) Primary ID of Target Chain 1"
TREMBL_COLUMN = "UniProt (TrEMBL) Primary ID of Target Chain 1"
SMILES_COLUMN = "Ligand SMILES"
TARGET_NAME_COLUMN = "Target Name"
REACTANT_ID_COLUMN = "BindingDB Reactant_set_id"
MONOMER_ID_COLUMN = "BindingDB MonomerID"

REQUIRED_COLUMNS = [
    ORGANISM_COLUMN,
    CHAIN_COUNT_COLUMN,
    SEQUENCE_COLUMN,
    SWISSPROT_COLUMN,
    TREMBL_COLUMN,
    SMILES_COLUMN,
    TARGET_NAME_COLUMN,
    REACTANT_ID_COLUMN,
    MONOMER_ID_COLUMN,
    *ENDPOINT_COLUMNS.values(),
]

MEASUREMENT_RE = re.compile(
    r"^\s*(?P<relation><=|>=|=|<|>|~|≈)?\s*"
    r"(?P<value>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)
TARGET_EXCLUSION_PATTERNS = [
    re.compile(r"\b(mutant|mutation|variant|engineered)\b", re.I),
    re.compile(r"\b(truncated|fragment|construct|partial)\b", re.I),
    re.compile(r"(?:\[|\(|\b(?:aa|residues?)\s*)\d+\s*[-–:]\s*\d+(?:\]|\))?", re.I),
    re.compile(r"(?:\[|\b)[A-Z]\d{1,5}[A-Z](?:\]|\b)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="BindingDB_All_202609_tsv.zip or TSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--uniprot-cache", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--uniprot-batch-size", type=int, default=80)
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def normalize_sequence(value: str) -> str:
    return re.sub(r"[^A-Za-z]", "", value or "").upper()


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def parse_measurement(raw: str) -> tuple[str, float] | None:
    match = MEASUREMENT_RE.fullmatch(raw or "")
    if not match:
        return None
    relation = match.group("relation") or "="
    value = float(match.group("value"))
    if not math.isfinite(value) or value <= 0:
        return None
    return relation, value


def target_name_is_excluded(name: str) -> bool:
    return any(pattern.search(name or "") for pattern in TARGET_EXCLUSION_PATTERNS)


@contextmanager
def open_tsv(path: Path) -> Iterator[IO[bytes]]:
    if path.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(path)
        members = [name for name in archive.namelist() if name.lower().endswith((".tsv", ".txt"))]
        if len(members) != 1:
            archive.close()
            raise ValueError(f"Expected exactly one TSV/TXT file in {path}, found {len(members)}")
        handle = archive.open(members[0], "r")
        try:
            yield handle
        finally:
            handle.close()
            archive.close()
    else:
        with path.open("rb") as handle:
            yield handle


def load_uniprot_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["query_accession"]: row for row in csv.DictReader(handle, delimiter="\t")}


def save_uniprot_cache(path: Path, cache: dict[str, dict[str, str]]) -> None:
    fields = ["query_accession", "returned_accession", "organism_name", "taxonomy_id", "sequence", "status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for accession in sorted(cache):
            writer.writerow({field: cache[accession].get(field, "") for field in fields})


def fetch_uniprot_batch(accessions: list[str], timeout: int) -> dict[str, dict[str, str]]:
    query = " OR ".join(f"accession:{accession}" for accession in accessions)
    params = urllib.parse.urlencode(
        {
            "query": f"({query})",
            "format": "tsv",
            "fields": "accession,organism_name,organism_id,sequence",
            "size": max(100, len(accessions) * 2),
        }
    )
    request = urllib.request.Request(
        "https://rest.uniprot.org/uniprotkb/search?" + params,
        headers={"User-Agent": "BaCNet-BindingDB-curation/1.0"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_text = response.read().decode("utf-8")
            break
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == 4:
                raise RuntimeError(f"UniProt request failed after retries: {error}") from error
            time.sleep(2**attempt)

    found: dict[str, dict[str, str]] = {}
    for row in csv.DictReader(response_text.splitlines(), delimiter="\t"):
        accession = row.get("Entry", "").strip()
        if accession:
            found[accession] = {
                "query_accession": accession,
                "returned_accession": accession,
                "organism_name": row.get("Organism", ""),
                "taxonomy_id": row.get("Organism (ID)", ""),
                "sequence": normalize_sequence(row.get("Sequence", "")),
                "status": "found",
            }
    return found


def validate_uniprot(
    accessions: set[str], cache_path: Path, batch_size: int, timeout: int
) -> dict[str, dict[str, str]]:
    cache = load_uniprot_cache(cache_path)
    missing = sorted(accessions.difference(cache))
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        found = fetch_uniprot_batch(batch, timeout)
        for accession in batch:
            cache[accession] = found.get(
                accession,
                {
                    "query_accession": accession,
                    "returned_accession": "",
                    "organism_name": "",
                    "taxonomy_id": "",
                    "sequence": "",
                    "status": "not_found",
                },
            )
        save_uniprot_cache(cache_path, cache)
    return cache


def read_candidates(input_path: Path, chunk_size: int) -> tuple[list[dict[str, object]], Counter]:
    candidates: list[dict[str, object]] = []
    counts: Counter = Counter()
    with open_tsv(input_path) as handle:
        chunks = pd.read_csv(
            handle,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=REQUIRED_COLUMNS,
            chunksize=chunk_size,
        )
        for chunk in chunks:
            counts["source_rows"] += len(chunk)
            chunk = chunk[chunk[ORGANISM_COLUMN].str.contains(SPECIES_NAME, case=False, regex=False)]
            counts["species_rows"] += len(chunk)
            for row in chunk.to_dict(orient="records"):
                if row[CHAIN_COUNT_COLUMN].strip() not in {"", "1"}:
                    counts["excluded_multichain"] += 1
                    continue
                if target_name_is_excluded(row[TARGET_NAME_COLUMN]):
                    counts["excluded_target_name"] += 1
                    continue
                accession = row[SWISSPROT_COLUMN].strip() or row[TREMBL_COLUMN].strip()
                if not accession or not re.fullmatch(r"[A-Z0-9][A-Z0-9-]{4,19}", accession):
                    counts["excluded_missing_uniprot"] += 1
                    continue
                sequence = normalize_sequence(row[SEQUENCE_COLUMN])
                if not sequence:
                    counts["excluded_missing_sequence"] += 1
                    continue
                canonical = canonicalize_smiles(row[SMILES_COLUMN])
                if canonical is None:
                    counts["excluded_invalid_smiles"] += 1
                    continue
                for endpoint, column in ENDPOINT_COLUMNS.items():
                    raw = row[column].strip()
                    if not raw:
                        continue
                    measurement = parse_measurement(raw)
                    if measurement is None:
                        counts["excluded_invalid_measurement"] += 1
                        continue
                    relation, value_nm = measurement
                    if relation != "=":
                        counts["excluded_non_exact_measurement"] += 1
                        continue
                    candidates.append(
                        {
                            "endpoint": endpoint,
                            "measurement_value_nm": value_nm,
                            "pAffinity": 9.0 - math.log10(value_nm),
                            "uniprot_accession": accession,
                            "protein_sequence": sequence,
                            "canonical_smiles": canonical,
                            "target_name": row[TARGET_NAME_COLUMN],
                            "BindingDB_Reactant_set_id": row[REACTANT_ID_COLUMN],
                            "BindingDB_MonomerID": row[MONOMER_ID_COLUMN],
                        }
                    )
    return candidates, counts


def curate_candidates(
    candidates: list[dict[str, object]], uniprot_cache: dict[str, dict[str, str]], counts: Counter
) -> pd.DataFrame:
    retained: list[dict[str, object]] = []
    for row in candidates:
        record = uniprot_cache.get(str(row["uniprot_accession"]), {})
        canonical_sequence = normalize_sequence(record.get("sequence", ""))
        if record.get("status") != "found" or not canonical_sequence:
            counts["excluded_uniprot_not_found"] += 1
            continue
        if row["protein_sequence"] != canonical_sequence:
            counts["excluded_sequence_mismatch"] += 1
            continue
        if len(canonical_sequence) > 1022:
            counts["excluded_sequence_too_long"] += 1
            continue
        retained.append(row)

    if not retained:
        return pd.DataFrame()
    frame = pd.DataFrame(retained)
    keys = ["endpoint", "uniprot_accession", "canonical_smiles"]
    aggregated = (
        frame.groupby(keys, as_index=False)
        .agg(
            pAffinity=("pAffinity", "median"),
            protein_sequence=("protein_sequence", "first"),
            target_name=("target_name", "first"),
            duplicate_measurement_count=("pAffinity", "size"),
            BindingDB_Reactant_set_id=("BindingDB_Reactant_set_id", lambda values: ";".join(sorted(set(map(str, values))))),
            BindingDB_MonomerID=("BindingDB_MonomerID", lambda values: ";".join(sorted(set(map(str, values))))),
        )
        .sort_values(keys)
    )
    aggregated["measurement_value_nm"] = 10.0 ** (9.0 - aggregated["pAffinity"])
    counts["retained_measurements"] = len(frame)
    counts["aggregated_pairs"] = len(aggregated)
    return aggregated


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.uniprot_cache or args.output_dir / "uniprot_validation_cache.tsv"
    candidates, counts = read_candidates(args.input, args.chunk_size)
    accessions = {str(row["uniprot_accession"]) for row in candidates}
    cache = validate_uniprot(accessions, cache_path, args.uniprot_batch_size, args.timeout)
    curated = curate_candidates(candidates, cache, counts)
    for endpoint in ENDPOINT_COLUMNS:
        output = args.output_dir / f"pseudomonas_aeruginosa_{endpoint.lower()}.csv"
        if curated.empty:
            pd.DataFrame().to_csv(output, index=False)
        else:
            curated[curated["endpoint"] == endpoint].to_csv(output, index=False)
        print(f"Wrote {output}")
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")


if __name__ == "__main__":
    main()
