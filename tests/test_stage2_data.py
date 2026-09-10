import unittest
from pathlib import Path
import tempfile
import subprocess
from unittest.mock import patch

from foliorecall.data_preparation import normalized_query, select_records, download_shard


class SelectionTests(unittest.TestCase):
    def test_connection_failure_reuses_piece_map(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "shard.parquet"
            control = Path(str(target) + ".aria2")
            calls = []
            def transfer(command, **kwargs):
                calls.append(command)
                if len(calls) == 1:
                    target.write_bytes(b"\0" * 32)
                    control.write_bytes(b"piece-map")
                    raise subprocess.CalledProcessError(1, command)
                self.assertTrue(control.exists())
                self.assertTrue(target.exists())
                target.write_bytes(b"x" * 32)
                control.unlink()
            with patch("foliorecall.data_preparation.shutil.which", return_value="aria2c"), patch(
                    "foliorecall.data_preparation.subprocess.run", side_effect=transfer), patch("foliorecall.data_preparation.time.sleep"):
                download_shard("https://example.test/shard", target, 32)
            self.assertEqual(len(calls), 2)

    def test_preallocated_file_is_not_treated_as_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "shard.parquet"
            target.write_bytes(b"\0" * 32)
            def complete(*args, **kwargs):
                self.assertFalse(target.exists())
                target.write_bytes(b"x" * 32)
            with patch("foliorecall.data_preparation.shutil.which", return_value="aria2c"), patch(
                    "foliorecall.data_preparation.subprocess.run", side_effect=complete) as run:
                self.assertEqual(download_shard("https://example.test/shard", target, 32), 32)
                self.assertEqual(download_shard("https://example.test/shard", target, 32), 0)
                self.assertEqual(run.call_count, 1)

    def test_pairing_and_duplicate_groups(self):
        rows = [{"id": str(i), "query": f"query {i}" if i < 30 else "",
                 "negatives": [str(j) for j in range(40) if j != i], "row_index": i} for i in range(40)]
        rows[0]["query"], rows[1]["query"] = "Ａ  B", "a b"
        rows[2]["negatives"] = ["missing", "2"] + rows[2]["negatives"]
        split, train, dev, info = select_records(rows, 8, 2, 8)
        self.assertEqual(normalized_query(" Ａ\tB "), "a b")
        self.assertEqual(set(info["excluded_duplicate_query_pages"]), {"0", "1"})
        self.assertFalse(train & dev)
        self.assertFalse((train | dev) & {"0", "1"})
        self.assertEqual(len(split["train"]), 8)
        for name, pool in (("train", train), ("dev", dev)):
            for row in split[name]:
                self.assertIn(row["positive"], pool)
                self.assertIn(row["negative"], pool)
                self.assertIn(row["negative"], rows[int(row["positive"])]["negatives"])
                self.assertEqual(row["negative"], row["valid_negative_ids"][0])
        self.assertEqual(select_records(rows, 8, 2, 8)[0], split)


if __name__ == "__main__":
    unittest.main()
