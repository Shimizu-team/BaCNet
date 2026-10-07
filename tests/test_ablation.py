import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ablation_models import ABLATION_VARIANTS, MODALITY_DIMS, create_ablation_model
from training_data import load_training_inputs_from_splits


class AblationTest(unittest.TestCase):
    def test_all_variants_accept_the_documented_input(self):
        inputs = torch.randn(2, sum(MODALITY_DIMS.values()))
        for variant in ABLATION_VARIANTS:
            with self.subTest(variant=variant):
                model = create_ablation_model(
                    variant,
                    projection_dim=8,
                    hidden_dims=(32, 16, 8),
                )
                model.eval()
                self.assertEqual(tuple(model(inputs).shape), (2, 1))

    def test_split_files_are_loaded_directly_with_column_mapping(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            rows = {
                "train": ("P1", "C1", 0.1),
                "validation": ("P2", "C2", 0.2),
                "test": ("P3", "C3", 0.3),
            }
            for split, row in rows.items():
                pd.DataFrame([row], columns=["ProteinID", "ChemID", "transformed_score"]).to_csv(
                    directory / f"{split}.csv", index=False
                )

            def save_embeddings(name, identifiers, dimension):
                path = directory / name
                torch.save({identifier: torch.zeros(dimension) for identifier in identifiers}, path)
                return path

            protein = save_embeddings("protein.pt", ["P1", "P2", "P3"], 5120)
            morgan = save_embeddings("morgan.pt", ["C1", "C2", "C3"], 1024)
            chemical_checker = save_embeddings("cc.pt", ["C1", "C2", "C3"], 1280)
            chemberta = save_embeddings("chemberta.pt", ["C1", "C2", "C3"], 384)
            inputs = load_training_inputs_from_splits(
                directory / "train.csv",
                directory / "validation.csv",
                directory / "test.csv",
                protein,
                morgan,
                chemical_checker,
                chemberta,
                column_names={
                    "protein_id": "ProteinID",
                    "compound_id": "ChemID",
                    "transformed_score": "transformed_score",
                },
            )
            self.assertEqual(inputs.links["split"].tolist(), ["train", "validation", "test"])
            self.assertEqual(inputs.links["transformed_score"].tolist(), [0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
