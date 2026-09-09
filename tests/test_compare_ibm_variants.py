"""Pruebas de los cruces de Accounts y del perfilado ampliado."""

import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from scripts.compare_datasets import HEADERS
from scripts.compare_ibm_variants import ACCOUNT_HEADER, audit_endpoint, load_accounts, normalize_bank, profile_variant, run_variants


class VariantTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.accounts = self.root / "HI-Small_accounts.csv"
        self.trans = self.root / "HI-Small_Trans.csv"

    def write(self, path, header, records):
        with path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target)
            writer.writerow(header)
            writer.writerows(records)

    def test_bank_normalization_preserves_account_identity(self):
        self.assertEqual(normalize_bank("03208"), "3208")
        self.write(self.accounts, ACCOUNT_HEADER, [
            ["Bank", "3208", "001", "E1", "Entity"],
            ["Bank", "3208", "1", "E2", "Other"],
        ])
        catalog, summary = load_accounts(self.accounts)
        audit = audit_endpoint(Counter({("03208", "001"): 4, ("3208", "1"): 1}), catalog, include_entities=True)
        self.assertEqual(summary["entities"], 2)
        self.assertEqual(audit["matched_transactions_literal"], 1)
        self.assertEqual(audit["matched_transactions_normalized"], 5)
        self.assertEqual(audit["active_entities"], 2)

    def test_conflicting_entities_are_not_silently_joined(self):
        self.write(self.accounts, ACCOUNT_HEADER, [
            ["Bank", "01", "A", "E1", "Entity"],
            ["Bank", "1", "A", "E2", "Other"],
        ])
        catalog, summary = load_accounts(self.accounts)
        audit = audit_endpoint(Counter({("01", "A"): 3}), catalog)
        self.assertEqual(summary["ambiguous_keys"], 1)
        self.assertEqual(audit["matched_transactions_normalized"], 0)
        self.assertEqual(audit["ambiguous_observed_accounts"], 1)

    def test_entity_counts_deduplicate_raw_bank_aliases(self):
        self.write(self.accounts, ACCOUNT_HEADER, [
            ["Bank", "1", "A", "E1", "Entity"],
            ["Bank", "2", "B", "E1", "Entity"],
        ])
        catalog, _ = load_accounts(self.accounts)
        audit = audit_endpoint(Counter({("01", "A"): 2, ("1", "A"): 1, ("2", "B"): 4}), catalog, include_entities=True)
        self.assertEqual(audit["raw_sender_aliases_after_bank_normalization"], 1)
        self.assertEqual(audit["active_entities_multiple_sending_accounts"], 1)
        self.assertEqual(audit["sending_accounts_in_multiaccount_entities"], 2)

    def test_coverage_is_weighted_by_transactions(self):
        audit = audit_endpoint(Counter({("1", "A"): 9, ("1", "MISSING"): 1}), {("1", "A"): ("E", "1")})
        self.assertEqual(audit["transaction_coverage_pct"], 90)
        self.assertEqual(audit["account_coverage_pct"], 50)

    def test_profile_reports_full_histories_and_both_endpoint_joins(self):
        self.write(self.accounts, ACCOUNT_HEADER, [
            ["Bank", "1", "A", "E1", "Entity"],
            ["Bank", "2", "B", "E1", "Entity"],
        ])
        events = [
            ["2022/09/02 00:00", "01", "A", "2", "B", "1", "USD", "1", "USD", "Wire", "0"],
            ["2022/09/01 00:00", "01", "A", "2", "B", "1", "USD", "1", "USD", "Wire", "1"],
            ["2022/09/03 00:00", "2", "B", "01", "A", "1", "USD", "1", "USD", "Wire", "0"],
        ]
        self.write(self.trans, HEADERS["ibm"], events)
        result = profile_variant(self.trans, self.accounts, progress_every=0)
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["positive_senders"], 1)
        self.assertEqual(result["time_min"], "2022/09/01 00:00")
        self.assertEqual(result["accounts"]["entities_multiple_accounts"], 1)
        self.assertEqual(result["origin_join"]["active_entities_multiple_sending_accounts"], 1)
        self.assertEqual(result["destination_join"]["transaction_coverage_pct"], 100)

    def test_cache_reuses_complete_results_and_invalidates_modified_input(self):
        self.write(self.accounts, ACCOUNT_HEADER, [["Bank", "1", "A", "E", "Entity"]])
        event = ["2022/09/01 00:00", "01", "A", "01", "A", "1", "USD", "1", "USD", "Wire", "0"]
        self.write(self.trans, HEADERS["ibm"], [event])
        output = self.root / "results"
        first = run_variants(self.root, output, variants=["HI-Small"])
        with patch("scripts.compare_ibm_variants.profile_variant", side_effect=AssertionError("No debe releer")):
            second = run_variants(self.root, output, variants=["HI-Small"])
        self.assertEqual(first[0]["rows"], second[0]["rows"])
        self.write(self.trans, HEADERS["ibm"], [event, event])
        third = run_variants(self.root, output, variants=["HI-Small"])
        self.assertEqual(third[0]["rows"], 2)


if __name__ == "__main__":
    unittest.main()
