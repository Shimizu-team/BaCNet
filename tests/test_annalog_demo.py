import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "annalog_demo"
EXPECTED_DIMENSIONS = {
    "morgan_fingerprint.pt": 1024,
    "chemical_checker.pt": 1280,
    "chemberta-2.pt": 384,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TestBasicScreeningAssets(unittest.TestCase):
    def test_mianserin_input_matches_precomputed_embedding_ids(self):
        compounds = pd.read_csv(
            ROOT / "examples" / "example_mols.csv",
            dtype={"Compound_ID": str},
        )
        self.assertEqual(
            compounds.to_dict("records"),
            [
                {
                    "Compound_ID": "Mianserin",
                    "SMILES": "CN1CCN2C(C1)C1=CC=CC=C1CC1=CC=CC=C21",
                }
            ],
        )

        expected_ids = compounds["Compound_ID"].tolist()
        library = ROOT / "examples" / "chemical_library"
        for filename, dimension in EXPECTED_DIMENSIONS.items():
            embeddings = torch.load(
                library / filename,
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(list(embeddings), expected_ids)
            self.assertEqual(tuple(embeddings["Mianserin"].shape), (dimension,))
            self.assertEqual(embeddings["Mianserin"].dtype, torch.float32)
            self.assertTrue(torch.isfinite(embeddings["Mianserin"]).all())


class TestANNalogDemoAssets(unittest.TestCase):
    def test_assets_and_manifest(self):
        compounds = pd.read_csv(DEMO / "compounds.csv", dtype={"Compound_ID": str})
        self.assertEqual(len(compounds), 30)
        self.assertEqual(compounds["Compound_ID"].nunique(), 30)
        self.assertTrue((compounds["generation_class"] == "medium").all())
        self.assertTrue((compounds["SA_score"] <= 5.0).all())
        self.assertTrue((compounds["QED"] >= 0.5).all())
        self.assertFalse(compounds["PAINS"].astype(str).str.lower().isin({"true", "1"}).any())
        expected_ids = compounds["Compound_ID"].tolist()

        for filename, dimension in EXPECTED_DIMENSIONS.items():
            embeddings = torch.load(
                DEMO / "embeddings" / filename, map_location="cpu", weights_only=True
            )
            self.assertEqual(list(embeddings), expected_ids)
            for vector in embeddings.values():
                self.assertEqual(vector.dtype, torch.float32)
                self.assertEqual(tuple(vector.shape), (dimension,))
                self.assertTrue(torch.isfinite(vector).all())

        manifest = json.loads((DEMO / "demo_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["compound_count"], 30)
        self.assertEqual(manifest["compound_ids"], expected_ids)
        for relative_path, expected_hash in manifest["outputs"].items():
            self.assertEqual(sha256(DEMO / relative_path), expected_hash)


class TestANNalogDemoInference(unittest.TestCase):
    def test_first_20_and_all_30_scores_match_full_library_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src" / "search_drug.py"),
                    "--model",
                    str(ROOT / "models" / "checkpoint_ecoli.pt"),
                    "--ecdf",
                    str(ROOT / "models" / "ecdf_bacnet_v1.npz"),
                    "--protein",
                    str(ROOT / "examples" / "target_protein" / "PBP_ecoli.pt"),
                    "--chemical",
                    str(DEMO / "embeddings"),
                    "--output",
                    temp_dir,
                ],
                cwd=ROOT,
                check=True,
            )
            actual = pd.read_csv(Path(temp_dir) / "P02918_screening_score.csv", dtype={"Compound_ID": str})

        expected = pd.read_csv(
            DEMO / "expected" / "P02918_screening_score.csv", dtype={"Compound_ID": str}
        )
        compounds = pd.read_csv(DEMO / "compounds.csv", dtype={"Compound_ID": str})
        comparison = compounds[["source_row", "Compound_ID"]].merge(
            actual, on="Compound_ID", validate="one_to_one"
        ).merge(
            expected[["Compound_ID", "CPI_score"]],
            on="Compound_ID",
            validate="one_to_one",
            suffixes=("_actual", "_expected"),
        ).sort_values("source_row")

        np.testing.assert_allclose(
            comparison.iloc[:20]["CPI_score_actual"],
            comparison.iloc[:20]["CPI_score_expected"],
            rtol=0,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            comparison["CPI_score_actual"],
            comparison["CPI_score_expected"],
            rtol=0,
            atol=1e-8,
        )


if __name__ == "__main__":
    unittest.main()
