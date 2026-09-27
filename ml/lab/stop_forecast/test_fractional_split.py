"""Synthetic fixtures check software only; they are not Moscow training evidence."""
import unittest
import numpy as np
from scipy.optimize import check_grad
from fractional_split import entropy_projection, fit, loss, predict


class FractionalTest(unittest.TestCase):
    def test_gradient_fit_and_no_fabricated_labels(self):
        x = np.array([[-1., 0], [0, 1], [1, 0], [-1, 1], [0, 0], [1, 1]])
        edges = np.array([0, 3, 6]); y = np.array([0., 3, 7, 0, 4, 6]); q = np.ones(6)
        args = (x, edges, y, q, 1., .01)
        self.assertLess(check_grad(lambda b: loss(b, *args)[0], lambda b: loss(b, *args)[1],
                                   np.linspace(-.3, .3, 5)), 1e-6)
        metadata = dict(observation_status=['observed_complete']*6,
                        evidence_ref=['synthetic_unit_fixture_not_research']*6,
                        route_totals=[10, 10], expected_sizes=[3, 3])
        model = fit(x, edges, y, q, **metadata)
        out = predict(model, x, edges, [10, 10])
        np.testing.assert_allclose(np.add.reduceat(out['boardings'], edges[:-1])+out['unassigned'], 10)
        for status in ['missing', 'observed_partial', 'restored_estimate', 'observed_unknown_completeness']:
            with self.assertRaises(ValueError):
                fit(x, edges, y, q, **{**metadata, 'observation_status': [status]*6})
        for fields in [dict(expected_sizes=[4, 3]), dict(route_totals=[11, 10]), dict(evidence_ref=['']*6)]:
            with self.assertRaises(ValueError):
                fit(x, edges, y, q, **{**metadata, **fields})
        no_stops = {'theta': np.array([-100., 0, 0, 0, 0])}
        out = predict(no_stops, x, edges, [10, 10])
        np.testing.assert_array_equal(out['unassigned'], [10, 10])
        np.testing.assert_array_equal(out['boardings'], 0)
        # Complete observed zero groups train BL only; no 0/0 fractions.
        self.assertTrue(np.isfinite(loss(np.zeros(5), x, edges, y*0, q, 1., .01)[0]))

    def test_training_entropy_penalty(self):
        x = np.array([[-1.], [1.], [-1.], [1.]])
        edges = np.array([0, 2, 4]); y = np.array([1., 9., 1., 9.])
        metadata = dict(observation_status=['observed_complete']*4,
                        evidence_ref=['synthetic_unit_fixture_not_research']*4,
                        route_totals=[10, 10], expected_sizes=[2, 2])
        weak = fit(x, edges, y, np.ones(4), strength=0, **metadata)
        strong = fit(x, edges, y, np.ones(4), strength=100, **metadata)
        self.assertLess(abs(strong['theta'][-1]), abs(weak['theta'][-1]))
        # Finite very large prior weights must not overflow normalization.
        self.assertTrue(np.isfinite(loss(np.zeros(3), x, edges, y,
                                       np.full(4, 1e308), 1., .01)[0]))

    def test_entropy_and_support(self):
        edges = np.array([0, 3, 5]); logits = np.array([-4., 0, 4, -1, 1]); q = np.ones(5)
        entropies = []
        for strength in [0., .1, 1., 10., 100.]:
            p = entropy_projection(logits, edges, q, strength)
            np.testing.assert_allclose(np.add.reduceat(p, edges[:-1]), 1)
            entropies.append(-np.sum(p*np.log(p)))
        self.assertTrue((np.diff(entropies) > 0).all())
        prior = np.array([1., 2, 3, 4, 1])
        p = entropy_projection(logits, edges, prior, 1e10)
        np.testing.assert_allclose(p, prior/np.repeat(np.add.reduceat(prior, edges[:-1]), np.diff(edges)), atol=1e-9)
        for bad in [np.zeros(5), np.array([1, 1, np.nan, 1, 1])]:
            with self.assertRaises(ValueError):
                entropy_projection(logits, edges, bad, 1)
        # KL projection satisfies its constrained first-order condition.
        p = entropy_projection(logits, edges, prior, 2)
        derivative = 3*np.log(p)-logits-2*np.log(prior)
        self.assertLess(np.ptp(derivative[:3]), 1e-12)
        self.assertLess(np.ptp(derivative[3:]), 1e-12)


if __name__ == '__main__':
    unittest.main()
