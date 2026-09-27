"""Check sparse inference, missingness, tempo transitions and trace construction."""
import unittest
import numpy as np
import pandas as pd

from activity_model import PRIOR
from build_elastic_data import blocks, select
from duration_model import duration_transition, infer
from elastic_model import geographic_chain, filter_intensity, observation_mask, predict, scores


class ElasticTest(unittest.TestCase):
    def test_forward_matches_existing_exact_duration_kernel(self):
        duration = np.array([[.4, .6], [.7, .3]])
        transition, survival = duration_transition([[0, 1], [1, 0]], duration)
        initial = (survival / survival.sum()).ravel()
        factors = np.repeat([.2, 3.], 2)
        counts = np.array([0, 2, 4, 0])
        posterior, expected = infer(counts[:, None], [[.2], [3]], [[0, 1], [1, 0]], duration,
                                    survival.sum(1) / survival.sum())
        _, terminal, actual = filter_intensity(counts, (transition, initial, factors), np.ones(4, bool), 1.)
        np.testing.assert_allclose(terminal.reshape(2, 2).sum(1), posterior[-1], atol=1e-12)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_stationarity_speed_switches_and_hidden_future(self):
        lengths = [200., 400., 700., 900., 300.]
        chain, info = geographic_chain(lengths, [0, 0, 0, 0, 1], [1, 1, 0, 1, 1], PRIOR, .9)
        transition, prior, factors = chain
        np.testing.assert_allclose(np.asarray(transition.sum(1)).ravel(), 1.)
        np.testing.assert_allclose(transition.T @ prior, prior, atol=1e-14)
        self.assertAlmostEqual(prior @ factors, 1.)
        self.assertEqual(info['states'], 45)
        self.assertGreater(transition[2, 12], 0)  # travel-2 of stop0/speed0 → stop1/speed1
        counts = np.random.default_rng(4).poisson(.7, 1080)
        first = predict(counts, chain)
        altered = counts.astype(float)
        altered[~observation_mask()] = np.nan
        second = predict(altered, chain)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
            self.assertAlmostEqual(a.sum(), 1.)
        altered = counts.copy()
        altered[372:] += 100
        np.testing.assert_array_equal(first[0], predict(altered, chain)[0])
        with self.assertRaises(ValueError):
            predict(-np.ones(1080), chain)
        with self.assertRaises(ValueError):
            geographic_chain(lengths, [0] * 5, [1] * 5, PRIOR, np.nan)
        self.assertEqual(scores(counts, first)['short_count'], counts[~observation_mask()][:60].sum())

    def test_uniform_geography_keeps_absolute_phase_ambiguous(self):
        chain, _ = geographic_chain([400.] * 4, [0] * 4, [1] * 4, PRIOR, .9)
        counts = np.random.default_rng(3).poisson(1., 100)
        _, end, _ = filter_intensity(counts, chain, np.ones(100, bool), 1.)
        np.testing.assert_allclose(end.reshape(4, 3, 3).sum((1, 2)), .25, atol=1e-12)

    def test_gaps_split_blocks_and_sampling_does_not_read_held(self):
        seconds = np.arange(0, 22000, 60)
        self.assertEqual(blocks(seconds), [0, 1080])
        split = seconds[(seconds < 6000) | (seconds >= 7200)]
        self.assertEqual(blocks(split), [720])
        self.assertEqual(blocks([]), [])
        f = pd.DataFrame({'date': ['2025-09-15'] * 12 + ['2025-09-23'] * 12,
                          'vehicle_day': list(range(6)) * 4, 'held_events': range(24)})
        selected = select(f)
        f['held_events'] = 0
        np.testing.assert_array_equal(selected, select(f))
        self.assertFalse(f.loc[selected].duplicated(['date', 'vehicle_day']).any())


if __name__ == '__main__':
    unittest.main()
