import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assemble_annalog_embeddings import main as assemble_main
from prepare_annalog_compounds import prepare
from run_annalog_bacnet import merge_rankings


class ANNalogBaCNetPipelineTest(unittest.TestCase):
    def test_prepare_canonicalizes_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            medium = root / "medium.csv"
            far = root / "far.csv"
            pd.DataFrame(
                [
                    {"SMILES": "C(C)O", "Score": -2.0},
                    {"SMILES": "CCO", "Score": -1.0},
                    {"SMILES": "not-a-smiles", "Score": -3.0},
                ]
            ).to_csv(medium, index=False)
            pd.DataFrame([{"SMILES": "c1ccccc1", "Score": -4.0}]).to_csv(far, index=False)
            prepared, rejected, counts = prepare(medium, far)
            self.assertEqual(len(prepared), 2)
            self.assertEqual(len(rejected), 1)
            self.assertEqual(counts["duplicates_removed"], 1)
            ethanol = prepared.loc[prepared["SMILES"] == "CCO"].iloc[0]
            self.assertEqual(float(ethanol["ANNalog_score"]), -1.0)
            self.assertTrue(str(ethanol["Compound_ID"]).startswith("annalog_"))

    def test_merge_rankings_preserves_metadata_and_sorts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compounds = root / "compounds.csv"
            scores = root / "scores.csv"
            output = root / "ranked.csv"
            pd.DataFrame(
                [
                    {"Compound_ID": "a", "SMILES": "CC", "ANNalog_score": -2.0},
                    {"Compound_ID": "b", "SMILES": "CCC", "ANNalog_score": -1.0},
                ]
            ).to_csv(compounds, index=False)
            pd.DataFrame(
                [
                    {"Compound_ID": "a", "CPI_score": 0.2},
                    {"Compound_ID": "b", "CPI_score": 0.8},
                ]
            ).to_csv(scores, index=False)
            self.assertEqual(merge_rankings(compounds, scores, output), 2)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["Compound_ID"], "b")
            self.assertEqual(rows[0]["BaCNet_rank"], "1")

    def test_assembly_keeps_only_compounds_with_all_valid_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compounds = root / "compounds.csv"
            output = root / "assembled"
            pd.DataFrame(
                [
                    {"Compound_ID": "a", "SMILES": "CC"},
                    {"Compound_ID": "b", "SMILES": "CCC"},
                ]
            ).to_csv(compounds, index=False)
            morgan = root / "morgan.pt"
            cc = root / "cc.pt"
            chemberta = root / "chemberta.pt"
            torch.save({"a": torch.zeros(1024), "b": torch.zeros(1024)}, morgan)
            torch.save({"a": torch.zeros(1280), "b": torch.zeros(1280)}, cc)
            torch.save({"a": torch.zeros(384)}, chemberta)
            argv = [
                "assemble_annalog_embeddings.py",
                "--compounds", str(compounds),
                "--morgan", str(morgan),
                "--chemical-checker", str(cc),
                "--chemberta", str(chemberta),
                "--output-dir", str(output),
            ]
            with patch("sys.argv", argv):
                assemble_main()
            scoring = pd.read_csv(output / "scoring_compounds.csv", dtype=str)
            failures = pd.read_csv(output / "embedding_failures.csv", dtype=str)
            self.assertEqual(scoring["Compound_ID"].tolist(), ["a"])
            self.assertIn("b", failures["Compound_ID"].tolist())


if __name__ == "__main__":
    unittest.main()
