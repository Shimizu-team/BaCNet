#!/usr/bin/env python3
"""Prepare the approved-compound DrugBank library used by BaCNet.

DrugBank data must be obtained separately under an appropriate DrugBank
license. This script accepts a DrugBank 5.1.12 XML file (plain XML or a ZIP
containing one XML file), retains records whose groups include ``approved``,
canonicalizes their supplied SMILES with RDKit, and removes PAINS A/B/C hits.

Canonicalization deliberately does not perform desalting, neutralization, or
tautomer standardization.
"""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator
from xml.etree import ElementTree

from rdkit import Chem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="DrugBank 5.1.12 XML or ZIP file")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    return parser.parse_args()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def canonicalize_smiles(smiles: str) -> str | None:
    """Return an isomeric canonical SMILES without additional standardization."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def build_pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog(params)


@contextmanager
def open_xml(path: Path) -> Iterator[BinaryIO]:
    if path.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(path)
        members = [name for name in archive.namelist() if name.lower().endswith(".xml")]
        if len(members) != 1:
            archive.close()
            raise ValueError(f"Expected exactly one XML file in {path}, found {len(members)}")
        handle = archive.open(members[0], "r")
        try:
            yield handle
        finally:
            handle.close()
            archive.close()
    else:
        with path.open("rb") as handle:
            yield handle


def direct_children(element: ElementTree.Element, name: str) -> list[ElementTree.Element]:
    return [child for child in element if local_name(child.tag) == name]


def child_text(element: ElementTree.Element, name: str) -> str:
    for child in element:
        if local_name(child.tag) == name:
            return (child.text or "").strip()
    return ""


def extract_drugbank_id(drug: ElementTree.Element) -> str:
    identifiers = direct_children(drug, "drugbank-id")
    for identifier in identifiers:
        if identifier.attrib.get("primary", "").lower() == "true":
            return (identifier.text or "").strip()
    return (identifiers[0].text or "").strip() if identifiers else ""


def extract_groups(drug: ElementTree.Element) -> set[str]:
    groups: set[str] = set()
    for container in direct_children(drug, "groups"):
        groups.update(
            (group.text or "").strip().lower()
            for group in direct_children(container, "group")
            if (group.text or "").strip()
        )
    return groups


def extract_smiles(drug: ElementTree.Element) -> str:
    accepted_kinds = {"smiles", "canonical smiles"}
    for container in direct_children(drug, "calculated-properties"):
        for prop in direct_children(container, "property"):
            if child_text(prop, "kind").lower() in accepted_kinds:
                value = child_text(prop, "value")
                if value:
                    return value
    return ""


def prepare_drugbank(input_path: Path, output_path: Path) -> dict[str, int]:
    catalog = build_pains_catalog()
    counts = {
        "records": 0,
        "approved": 0,
        "missing_smiles": 0,
        "invalid_smiles": 0,
        "pains": 0,
        "retained": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=["Compound_ID", "SMILES"])
        writer.writeheader()
        with open_xml(input_path) as input_handle:
            for _event, element in ElementTree.iterparse(input_handle, events=("end",)):
                if local_name(element.tag) != "drug":
                    continue
                counts["records"] += 1
                if "approved" not in extract_groups(element):
                    element.clear()
                    continue
                counts["approved"] += 1
                smiles = extract_smiles(element)
                if not smiles:
                    counts["missing_smiles"] += 1
                    element.clear()
                    continue
                canonical = canonicalize_smiles(smiles)
                if canonical is None:
                    counts["invalid_smiles"] += 1
                    element.clear()
                    continue
                mol = Chem.MolFromSmiles(canonical)
                if mol is None:
                    counts["invalid_smiles"] += 1
                    element.clear()
                    continue
                if catalog.HasMatch(mol):
                    counts["pains"] += 1
                    element.clear()
                    continue
                compound_id = extract_drugbank_id(element)
                if not compound_id:
                    print("Warning: approved record without a DrugBank ID was skipped", file=sys.stderr)
                    element.clear()
                    continue
                writer.writerow({"Compound_ID": compound_id, "SMILES": canonical})
                counts["retained"] += 1
                element.clear()
    return counts


def main() -> None:
    args = parse_args()
    counts = prepare_drugbank(args.input, args.output)
    for key, value in counts.items():
        print(f"{key}: {value}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
