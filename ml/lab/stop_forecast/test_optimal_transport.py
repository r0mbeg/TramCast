"""Small independent checks of the OT pilot, not Moscow accuracy tests."""
import itertools
import unittest

import numpy as np
from optimal_transport import partial_cost, prefix_peaks, predict, templates


class OptimalTransportTest(unittest.TestCase):
    def test_transport_against_exhaustive_partial_matchings(self):
        a, b = np.array([0., 45., 200.]), np.array([3., 50.])
        values = []
        for assignment in itertools.product(range(-1, len(b)), repeat=len(a)):
            used = [j for j in assignment if j >= 0]
            if len(used) != len(set(used)):
                continue
            values.append(sum(((a[i] - b[j]) / 30) ** 2 for i, j in enumerate(assignment) if j >= 0)
                          + 4 * (len(a) - len(used)) + .25 * (len(b) - len(used)))
        self.assertAlmostEqual(partial_cost(a, b)[0], min(values))
        self.assertEqual(partial_cost([], b), (.5, 0))
        self.assertEqual(partial_cost(a, []), (12., 0))
        self.assertEqual(partial_cost([0], [1000]), (4.25, 0))
        with self.assertRaises(ValueError):
            partial_cost([np.nan], [0])

    def test_hidden_future_and_cyclic_relabelling(self):
        counts = np.zeros(360)
        counts[np.arange(10, 230, 20)] = 10
        peaks = prefix_peaks(counts)
        changed = counts.copy()
        changed[240:] = 1000000
        np.testing.assert_array_equal(peaks, prefix_peaks(changed))
        lengths = np.array([400., 700., 200., 900., 300.])
        turns = np.array([False, False, False, False, True])
        boarding = np.array([True, True, False, True, True])
        p, _ = predict(peaks, templates(lengths, turns, boarding))
        shifted, _ = predict(peaks, templates(np.roll(lengths, 2), np.roll(turns, 2), np.roll(boarding, 2)))
        np.testing.assert_allclose(p, shifted, atol=1e-12)
        self.assertAlmostEqual(p.sum(), 1.)
        self.assertTrue(np.isfinite(p).all() and (p >= .1 / 120 - 1e-15).all())

    def test_known_synthetic_template_predicts_its_continuation(self):
        visits = np.arange(100., 3600., 200.)
        # First template continues the prefix; second loses phase after the cutoff.
        wrong = visits.copy()
        wrong[wrong >= 2400] += 100
        p, _ = predict(visits[visits <= 2370], [(visits - 100, 0, 15, 0)])
        q, _ = predict(visits[visits <= 2370], [(wrong - 100, 0, 15, 0)])
        bins = ((visits[visits >= 2400] - 2400) / 10).astype(int)
        self.assertGreater(np.log(p[bins]).sum(), np.log(q[bins]).sum())
        # Same fitted prefix cannot decide between these two futures: no false identification.
        mixture, diagnostic = predict(visits[visits <= 2370],
            [(visits - 100, 0, 15, 0), (wrong - 100, 1, 15, 0)])
        np.testing.assert_allclose(mixture, (p + q) / 2)
        self.assertAlmostEqual(diagnostic['max_hypothesis_weight'], .5)


if __name__ == '__main__':
    unittest.main()
