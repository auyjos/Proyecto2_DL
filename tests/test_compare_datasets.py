"""Casos pequeños que comprueban decisiones capaces de sesgar la comparación."""

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.compare_datasets import HEADERS, compare_ibm_sample, length_summary, profile_dataset


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "input.csv"

    def write_ibm(self, events):
        # events: timestamp, banco, cuenta, etiqueta.
        with self.path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target)
            writer.writerow(HEADERS["ibm"])
            for timestamp, bank, account, label in events:
                writer.writerow([timestamp, bank, account, "900", "DEST", "1", "US Dollar", "1", "US Dollar", "Wire", label])

    def test_sender_identity_preserves_bank_and_leading_zeroes(self):
        self.write_ibm([
            ("2022/09/01 00:00", "01", "001", 1),
            ("2022/09/01 00:01", "02", "001", 0),
            ("2022/09/01 00:02", "01", "1", 0),
        ])
        summary, counts = profile_dataset(self.path, "ibm")
        self.assertEqual(summary["senders"], 3)
        self.assertIn(("01", "001"), counts)
        self.assertEqual(summary["positive_senders"], 1)

    def test_crop_uses_time_not_file_order_and_relabels(self):
        self.write_ibm([
            ("2022/09/03 00:00", "01", "A", 0),
            ("2022/09/01 00:00", "01", "A", 1),
            ("2022/09/02 00:00", "01", "A", 0),
        ])
        _, counts = profile_dataset(self.path, "ibm")
        sample = compare_ibm_sample(self.path, counts, n_senders=1, max_len=2)
        self.assertEqual(sample["positive_senders_full"], 1)
        self.assertEqual(sample["positive_senders_retained"], 0)
        self.assertEqual(sample["retained_transactions"], 2)
        self.assertEqual(sample["truncated_senders"], 1)

    def test_tied_timestamps_use_original_row_index(self):
        self.write_ibm([
            ("2022/09/01 00:00", "01", "A", 1),
            ("2022/09/01 00:00", "01", "A", 0),
        ])
        _, counts = profile_dataset(self.path, "ibm")
        sample = compare_ibm_sample(self.path, counts, n_senders=1, max_len=1)
        self.assertEqual(sample["positive_senders_retained"], 0)

    def test_sample_is_independent_of_key_order(self):
        self.write_ibm([("2022/09/01 00:00", "01", str(i), i % 2) for i in range(10)])
        _, counts = profile_dataset(self.path, "ibm")
        first = compare_ibm_sample(self.path, counts, n_senders=4)
        second = compare_ibm_sample(self.path, dict(reversed(list(counts.items()))), n_senders=4)
        self.assertEqual(first, second)

    def test_quantile_is_nearest_rank_without_interpolation(self):
        result = length_summary({"a": 1, "b": 1, "c": 4, "d": 100})
        self.assertEqual(result["length_quantiles"]["0.5"], 1)
        self.assertEqual(result["length_quantiles"]["0.95"], 100)
        self.assertEqual(result["senders_ge_5"], 1)

    def test_invalid_sample_and_label_are_rejected(self):
        self.write_ibm([("2022/09/01 00:00", "01", "A", 0)])
        _, counts = profile_dataset(self.path, "ibm")
        with self.assertRaises(ValueError):
            compare_ibm_sample(self.path, counts, n_senders=2)
        self.write_ibm([("2022/09/01 00:00", "01", "A", 2)])
        with self.assertRaises(ValueError):
            profile_dataset(self.path, "ibm")


if __name__ == "__main__":
    unittest.main()

