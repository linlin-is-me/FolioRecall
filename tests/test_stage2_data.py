import unittest

from foliorecall.data_preparation import normalized_query, select_records


class SelectionTests(unittest.TestCase):
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
