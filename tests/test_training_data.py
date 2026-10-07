import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import BACNET_INPUT_DIM
from training_data import (
    BaCNetTrainingDataset,
    make_synthetic_inputs,
    reference_leave_one_protein_out_split,
    validate_training_inputs,
)


class TrainingDataTest(unittest.TestCase):
    def test_synthetic_inputs_are_valid_and_have_expected_dimension(self):
        inputs = make_synthetic_inputs()
        report = validate_training_inputs(inputs)
        self.assertTrue(report.empty, report.to_string())
        dataset = BaCNetTrainingDataset(inputs.links, inputs)
        features, target, _ = dataset[0]
        self.assertEqual(features.numel(), BACNET_INPUT_DIM)
        self.assertEqual(target.ndim, 0)

    def test_transformed_score_is_not_transformed_again(self):
        inputs = make_synthetic_inputs()
        dataset = BaCNetTrainingDataset(inputs.links, inputs)
        _, target, _ = dataset[3]
        self.assertAlmostEqual(float(target), float(inputs.links.iloc[3]["transformed_score"]))

    def test_reference_leave_one_protein_out_is_deterministic_and_disjoint(self):
        inputs = make_synthetic_inputs()
        links = inputs.links.drop(columns="split")
        first = reference_leave_one_protein_out_split(links)
        second = reference_leave_one_protein_out_split(links)
        self.assertEqual(first["split"].tolist(), second["split"].tolist())
        groups = {
            split: set(first.loc[first["split"] == split, "protein_id"])
            for split in ("train", "validation", "test")
        }
        self.assertTrue(groups["train"].isdisjoint(groups["validation"]))
        self.assertTrue(groups["train"].isdisjoint(groups["test"]))
        self.assertTrue(groups["validation"].isdisjoint(groups["test"]))

    def test_validation_rejects_protein_leakage(self):
        inputs = make_synthetic_inputs()
        inputs.links.loc[inputs.links["split"] == "test", "protein_id"] = inputs.links.iloc[0]["protein_id"]
        report = validate_training_inputs(inputs)
        self.assertTrue(report["error"].str.contains("leave-one-protein-out leakage").any())


if __name__ == "__main__":
    unittest.main()
