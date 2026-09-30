import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import BACNET_INPUT_DIM
from training_data import BaCNetTrainingDataset, make_synthetic_inputs, validate_training_inputs


class TrainingDataTest(unittest.TestCase):
    def test_synthetic_inputs_are_valid_and_have_expected_dimension(self):
        inputs = make_synthetic_inputs()
        report = validate_training_inputs(inputs)
        self.assertTrue(report.empty, report.to_string())
        dataset = BaCNetTrainingDataset(inputs.links, inputs)
        features, target, _ = dataset[0]
        self.assertEqual(features.numel(), BACNET_INPUT_DIM)
        self.assertEqual(target.ndim, 0)

    def test_target_bct_is_not_transformed(self):
        inputs = make_synthetic_inputs()
        dataset = BaCNetTrainingDataset(inputs.links, inputs)
        _, target, _ = dataset[3]
        self.assertAlmostEqual(float(target), float(inputs.links.iloc[3]["target_bct"]))


if __name__ == "__main__":
    unittest.main()
