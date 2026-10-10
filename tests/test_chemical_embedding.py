import contextlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemical_embedding import MorganFeaturizer, apply_filters, summarize_filtering


class ChemicalFilterBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "Compound_ID": ["boundary", "sa_fail", "qed_fail", "pains_fail"],
                "PAINS": [False, False, False, True],
                "SA_score": [5.0, 5.0001, 4.0, 4.0],
                "QED": [0.5, 0.6, 0.4999, 0.6],
            }
        )

    def test_filter_includes_exact_sa_and_qed_boundaries(self):
        filtered = apply_filters(self.frame)
        self.assertEqual(filtered["Compound_ID"].tolist(), ["boundary"])

    def test_summary_uses_the_same_inclusive_boundaries(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            summarize_filtering(self.frame)
        summary = output.getvalue()
        self.assertIn("PAINS matches: 1/4", summary)
        self.assertIn("SA_score <= 5.00: 3/4", summary)
        self.assertIn("QED >= 0.50: 3/4", summary)


class MorganInputRepresentationTest(unittest.TestCase):
    def test_equivalent_mianserin_smiles_give_the_same_fingerprint(self):
        featurizer = MorganFeaturizer(n_bits=1024, radius=2)
        source_smiles = "CN1CCN2C(C1)C1=CC=CC=C1CC1=CC=CC=C21"
        canonical_smiles = "CN1CCN2c3ccccc3Cc3ccccc3C2C1"

        source = featurizer.smiles_to_morgan(source_smiles)
        canonical = featurizer.smiles_to_morgan(canonical_smiles)

        np.testing.assert_array_equal(source, canonical)


if __name__ == "__main__":
    unittest.main()
