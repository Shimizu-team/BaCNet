import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import BACNET_INPUT_DIM
from training_data import (
    BaCNetTrainingDataset,
    create_dataloaders,
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

    def test_split_whitespace_is_normalized_before_loading(self):
        inputs = make_synthetic_inputs()
        inputs.links.loc[0, "split"] = " train "
        report = validate_training_inputs(inputs)
        self.assertTrue(report.empty, report.to_string())
        loaders = create_dataloaders(inputs, batch_size=4, num_workers=0, seed=123)
        self.assertEqual(sum(len(loader.dataset) for loader in loaders.values()), len(inputs.links))
        self.assertEqual(inputs.links.loc[0, "split"], "train")

    def test_duplicate_protein_compound_pair_is_rejected_even_with_pair_id(self):
        inputs = make_synthetic_inputs()
        duplicate = inputs.links.iloc[0].copy()
        duplicate["pair_id"] = "different_pair_id"
        inputs.links = pd.concat([inputs.links, pd.DataFrame([duplicate])], ignore_index=True)
        report = validate_training_inputs(inputs)
        self.assertTrue(report["error"].str.contains("duplicated training pair").any())

    def test_transformed_score_outside_unit_interval_is_rejected(self):
        inputs = make_synthetic_inputs()
        inputs.links.loc[0, "transformed_score"] = 1.2
        report = validate_training_inputs(inputs)
        self.assertTrue(report["error"].str.contains("normalized range").any())


if __name__ == "__main__":
    unittest.main()
