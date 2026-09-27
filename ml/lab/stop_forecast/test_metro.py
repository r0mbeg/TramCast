"""Metro covariates respect quarter cutoff and ambiguous/missing inputs."""
import unittest
import numpy as np
import pandas as pd
from metro_experiment import features


class MetroTest(unittest.TestCase):
    def test_frozen_covariates(self):
        stops = pd.DataFrame({'stop_name': ['Метро "А"', 'Метро "Б"', 'Метро "В"', 'Улица']})
        metro = pd.DataFrame(dict(metro_name_key=['а', 'б', 'в', 'в'],
            IncomingPassengers=[100, 200, 300, 400], OutgoingPassengers=[150, 250, 350, 450],
            period_end=['2025-03-31']*4))
        values, info = features(stops, metro, '2025-04-30')
        self.assertEqual(info['usable'], 2)
        self.assertEqual(info['ambiguous'], 1)
        np.testing.assert_array_equal(values[:, 1], [1, 1, 0, 0])
        np.testing.assert_array_equal(values[2:, 2:], 0)
        future = metro.assign(period_end='2025-06-30', IncomingPassengers=99999)
        newer, _ = features(stops, pd.concat([metro, future]), '2025-04-30')
        np.testing.assert_array_equal(values, newer)
        self.assertAlmostEqual(values[:2, 2].mean(), 0)
        with self.assertRaises(ValueError):
            features(stops, metro, '2025-01-31')


if __name__ == '__main__':
    unittest.main()
