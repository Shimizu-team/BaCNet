import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aggregate_annalog_results import aggregate_annalog_results


class ANNalogWorkflowTest(unittest.TestCase):
    def test_aggregate_retains_filter_failures_and_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                [
                    {
                        "Compound_ID": "1",
                        "SMILES": "CCO",
                        "Score": -1.0,
                        "PAINS": False,
                        "SA_score": 2.0,
                        "QED": 0.7,
                    },
                    {
                        "Compound_ID": "2",
                        "SMILES": "CCC",
                        "Score": -2.0,
                        "PAINS": True,
                        "SA_score": 6.0,
                        "QED": 0.4,
                    },
                ]
            ).to_csv(root / "filters.csv", index=False)
            pd.DataFrame(
                [{"Compound_ID": "tensor(1)", "CPI_score": 0.9}]
            ).to_csv(root / "scores.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "run_id": "run1",
                        "iteration": 1,
                        "parent_id": "parent",
                        "parent_smiles": "CC",
                        "generation_class": "medium",
                        "generation_method": "sampling",
                        "temperature": 1.2,
                        "seed": 42,
                        "filter_csv": "filters.csv",
                        "bacnet_csv": "scores.csv",
                        "included_in_figure": True,
                    }
                ]
            ).to_csv(root / "runs.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "source_run_id": "run1",
                        "compound_id": "1",
                        "selected_parent_id": "selected1",
                        "selection_basis": "high score plus structural assessment",
                        "structure_method": "test method",
                        "structure_assessment": "plausible pose",
                    }
                ]
            ).to_csv(root / "selection.csv", index=False)

            output = aggregate_annalog_results(
                root / "runs.csv",
                root,
                root / "combined.csv",
                root / "selection.csv",
                root / "summary.csv",
            )

            self.assertEqual(len(output), 2)
            passed = output.loc[output["compound_id"] == "1"].iloc[0]
            failed = output.loc[output["compound_id"] == "2"].iloc[0]
            self.assertTrue(bool(passed["filter_pass"]))
            self.assertEqual(float(passed["bacnet_score"]), 0.9)
            self.assertEqual(int(passed["bacnet_rank"]), 1)
            self.assertTrue(bool(passed["selected_for_next_iteration"]))
            self.assertFalse(bool(failed["filter_pass"]))
            self.assertEqual(failed["filter_failure_reason"], "PAINS;SA_score>5;QED<0.5")
            self.assertTrue(pd.isna(failed["bacnet_score"]))
            summary = pd.read_csv(root / "summary.csv")
            self.assertEqual(int(summary.loc[0, "generated_count"]), 2)
            self.assertEqual(int(summary.loc[0, "filter_pass_count"]), 1)
            self.assertEqual(int(summary.loc[0, "selected_parent_count"]), 1)


if __name__ == "__main__":
    unittest.main()
