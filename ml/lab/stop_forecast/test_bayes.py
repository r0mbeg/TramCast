"""Numerical checks for aggregate likelihood, gradient and unidentified shares."""
import unittest
import numpy as np
from scipy.optimize import check_grad
from bayes_experiment import moments, objective, curvature


class BayesTest(unittest.TestCase):
    def test_aggregate_likelihood(self):
        x = np.array([[1., 0, -1], [1, 0, 1], [0, 1, -2], [0, 1, 2]])
        edges = np.array([0, 2, 4]); beta = np.array([1., 2, .3])
        centre = np.zeros(3); precision = np.ones(3)
        n, y = np.array([2, 3]), np.array([7, 19])
        for kind in ['poisson', 'nb']:
            args = (x, edges, n, y, centre, precision, kind)
            self.assertLess(check_grad(lambda b: objective(b, *args)[0],
                                       lambda b: objective(b, *args)[1], beta), 1e-4)
            eps = 1e-5
            numeric = np.column_stack([(objective(beta+e, *args)[1]-objective(beta-e, *args)[1])/(2*eps)
                for e in np.eye(3)*eps])
            np.testing.assert_allclose(curvature(beta, *args), numeric, rtol=1e-7, atol=1e-7)
        logm, p, _ = moments(beta, x, edges)
        np.testing.assert_allclose(np.exp(logm), np.add.reduceat(np.exp(x@beta), edges[:-1]))
        np.testing.assert_allclose(np.add.reduceat(p, edges[:-1]), 1)
        other = beta.copy(); other[2] += 1
        after, _, _ = moments(other, x, edges)
        other[:2] += logm-after
        fixed, changed, _ = moments(other, x, edges)
        np.testing.assert_allclose(fixed, logm, atol=1e-12)
        self.assertGreater(abs(changed-p).sum(), .1)
        # Sufficient statistics preserve NB objective up to parameter-independent constants.
        expected = sum((yy+10)*np.log(10+np.exp(logm[g]))-yy*logm[g]
                       for g, ys in enumerate([[3, 4], [5, 6, 8]]) for yy in ys)
        actual = objective(beta, x, edges, n, y, centre, np.zeros(3), 'nb')[0]
        mean = y/n
        saturated = sum((yy+10)*np.log(10+mean[g])-yy*np.log(mean[g])
                        for g, ys in enumerate([[3, 4], [5, 6, 8]]) for yy in ys)
        self.assertAlmostEqual(actual, expected-saturated)


if __name__ == '__main__':
    unittest.main()
