import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_bindingdb import (
    canonicalize_smiles as canonicalize_bindingdb_smiles,
    curate_candidates,
    parse_measurement,
    target_name_is_excluded,
)
from prepare_drugbank import build_pains_catalog, prepare_drugbank


class BindingDBPreparationTest(unittest.TestCase):
    def test_measurements_require_positive_parseable_values(self):
        self.assertEqual(parse_measurement("=10"), ("=", 10.0))
        self.assertEqual(parse_measurement("< 3.5"), ("<", 3.5))
        self.assertIsNone(parse_measurement("0"))
        self.assertIsNone(parse_measurement("not reported"))

    def test_target_name_flags_variants_and_fragments(self):
        self.assertTrue(target_name_is_excluded("LasB A113G mutant"))
        self.assertTrue(target_name_is_excluded("protein fragment"))
        self.assertFalse(target_name_is_excluded("Elastase LasB"))

    def test_canonicalization_does_not_remove_salt_components(self):
        canonical = canonicalize_bindingdb_smiles("C[NH2+]C.[Cl-]")
        self.assertIsNotNone(canonical)
        self.assertIn(".", canonical)
        self.assertIn("[Cl-]", canonical)

    def test_duplicate_measurements_use_median_paffinity(self):
        candidates = [
            {
                "endpoint": "Ki",
                "measurement_value_nm": 10.0,
                "pAffinity": 8.0,
                "uniprot_accession": "P12345",
                "protein_sequence": "ACDE",
                "canonical_smiles": "CCO",
                "target_name": "Target",
                "BindingDB_Reactant_set_id": "1",
                "BindingDB_MonomerID": "10",
            },
            {
                "endpoint": "Ki",
                "measurement_value_nm": 1000.0,
                "pAffinity": 6.0,
                "uniprot_accession": "P12345",
                "protein_sequence": "ACDE",
                "canonical_smiles": "CCO",
                "target_name": "Target",
                "BindingDB_Reactant_set_id": "2",
                "BindingDB_MonomerID": "10",
            },
        ]
        cache = {"P12345": {"status": "found", "sequence": "ACDE"}}
        result = curate_candidates(candidates, cache, Counter())
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.iloc[0]["pAffinity"], 7.0)
        self.assertAlmostEqual(result.iloc[0]["measurement_value_nm"], 100.0)
        self.assertEqual(result.iloc[0]["duplicate_measurement_count"], 2)


class DrugBankPreparationTest(unittest.TestCase):
    def test_all_three_pains_catalogs_are_loaded(self):
        self.assertGreater(build_pains_catalog().GetNumEntries(), 0)

    def test_approved_group_and_missing_smiles_selection(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<drugbank xmlns="http://www.drugbank.ca">
  <drug>
    <drugbank-id primary="true">DB00001</drugbank-id>
    <groups><group>approved</group><group>investigational</group></groups>
    <calculated-properties><property><kind>SMILES</kind><value>C[NH2+]C.[Cl-]</value></property></calculated-properties>
  </drug>
  <drug>
    <drugbank-id primary="true">DB00002</drugbank-id>
    <groups><group>approved</group></groups>
    <calculated-properties />
  </drug>
  <drug>
    <drugbank-id primary="true">DB00003</drugbank-id>
    <groups><group>investigational</group></groups>
    <calculated-properties><property><kind>SMILES</kind><value>CCO</value></property></calculated-properties>
  </drug>
</drugbank>
"""
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "drugbank.xml"
            output_path = Path(directory) / "approved.csv"
            input_path.write_text(xml, encoding="utf-8")
            counts = prepare_drugbank(input_path, output_path)
            output = output_path.read_text(encoding="utf-8")
        self.assertEqual(counts["approved"], 2)
        self.assertEqual(counts["missing_smiles"], 1)
        self.assertEqual(counts["retained"], 1)
        self.assertIn("DB00001", output)
        self.assertIn("[Cl-]", output)
        self.assertNotIn("DB00003", output)


if __name__ == "__main__":
    unittest.main()
