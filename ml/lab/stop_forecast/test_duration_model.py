"""Independent exact enumeration, symmetry, boundary and missingness checks."""
from itertools import product
import numpy as np
from scipy.special import gammaln, xlogy
from duration_model import infer, duration_transition

a = np.array([[0, 1], [1, 0]])
d = np.array([[.4, .6], [.7, .3]])
rates = np.array([[.2], [3.]])
y = np.array([[0], [2], [4], [0]])
prior = np.array([.3, .7])
p, logz = infer(y, rates, a, d, prior)
transition, survival = duration_transition(a, d)
transition = transition.toarray()
initial = (prior[:, None]*survival/survival.sum(1, keepdims=True)).ravel()
e = np.repeat(np.exp((xlogy(y[:, None, :], rates[None])-rates[None]-gammaln(y[:, None, :]+1)).sum(2)), 2, axis=1)
weights = np.zeros_like(p); total = 0
for path in product(range(4), repeat=len(y)):
    w = initial[path[0]]*e[0, path[0]]
    for t in range(1, len(y)):
        w *= transition[path[t-1], path[t]]*e[t, path[t]]
    total += w
    for t, state in enumerate(path):
        weights[t, state//2] += w
np.testing.assert_allclose(p, weights/total, atol=1e-12)
np.testing.assert_allclose(logz, np.log(total), atol=1e-12)

ring = np.roll(np.eye(3), 1, axis=1)
duration = np.tile([0, 1.], (3, 1))
# No amount of equal-emission data may invent an absolute phase.
ambiguous, _ = infer(np.ones((60, 1)), np.ones((3, 1)), ring, duration, np.ones(3)/3)
np.testing.assert_allclose(ambiguous, 1/3, atol=1e-12)
# A supplied boundary anchor makes the deterministic trajectory known.
anchored, _ = infer(np.ones((12, 1)), np.ones((3, 1)), ring, duration, [1,0,0], start_at_entry=True)
np.testing.assert_array_equal(anchored.argmax(1), np.repeat([0,1,2,0,1,2], 2))
assert anchored.max(1).min()==1
missing, _ = infer(np.full((6, 1), np.nan), np.ones((3, 1)), ring, duration,
                   np.ones(3)/3, observed=np.zeros((6,1), bool))
np.testing.assert_allclose(missing, 1/3, atol=1e-12)
# Distinct emissions and genuinely random durations carry information beyond anchors.
rng = np.random.default_rng(20260927)
truth = []
state = 0
random_duration = np.tile([0, .2, .6, .2], (3,1))
while len(truth)<120:
    truth.extend([state]*int(rng.choice([1,2,3,4],p=random_duration[state])))
    state = (state+1)%3
truth = np.asarray(truth[:120])
signal_rates = np.array([[.1,.2],[10,8],[25,30.]])
counts = rng.poisson(signal_rates[truth])
recovered, _ = infer(counts, signal_rates, ring, random_duration, [1,0,0], start_at_entry=True)
assert (recovered.argmax(1)==truth).mean()>.9
try:
    infer([[-1]], rates, a, d, prior)
except ValueError:
    pass
else:
    raise AssertionError('Negative count accepted')
print('Duration inference: enumeration, unknown phase, anchor, censoring and missingness passed')
