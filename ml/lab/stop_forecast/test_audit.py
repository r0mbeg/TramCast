"""Small runnable regression check: python -m unittest discover -s ml/lab/stop_forecast."""
import unittest
import pandas as pd
from audit import aggregate, degree_summary, valid_key


class AuditTest(unittest.TestCase):
    def test_boundaries_and_ambiguous_identifiers(self):
        times = ["2025-01-01 00:59:59", "2025-01-01 01:00:00",
                 "2025-01-01 05:29:59", "2025-01-01 05:30:00",
                 "2025-01-01 05:30:00", "2025-11-01 00:00:00"]
        frame = pd.DataFrame(dict(tran_date_time=times, ngpt_route=["12 трамвай"] * 6,
                                  validation_result=[1, 1, 1, 1, 0, 1]))
        raw, clean, working, counts = aggregate(frame, "2025-01-01", "2025-11-01")
        self.assertEqual(int(raw.sum()), 4)
        self.assertEqual(int(clean.sum()), 2)
        self.assertEqual(int(working.sum()), 3)
        self.assertEqual(counts["outside_period"], 1)
        self.assertEqual(clean.to_dict(), {(12, "2025-01-01", 0): 1, (12, "2025-01-01", 5): 1})
        self.assertEqual(valid_key(pd.Series(["", "0", "NULL", "  ", "0012"])).tolist(),
                         [False, False, False, False, True])
        self.assertEqual(degree_summary([("a", "x"), ("a", "x"), ("a", "y"), ("b", "z")]),
                         dict(keys=2, keys_with_multiple_values=1, maximum_values=2))
        # A device changing vehicle across days need not be ambiguous on either day.
        self.assertEqual(degree_summary([(("01", "a"), "x"), (("02", "a"), "y")])["keys_with_multiple_values"], 0)
        with self.assertRaises(ValueError):
            aggregate(frame.assign(ngpt_route="unknown"), "2025-01-01", "2025-11-01")


if __name__ == "__main__":
    unittest.main()
