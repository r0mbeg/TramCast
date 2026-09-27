"""Independent BL–FS with supervised KL penalty; no route-only fitting."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def validate_layout(x, edges, q):
    x, edges, q = np.asarray(x, float), np.asarray(edges), np.asarray(q, float)
    if (x.ndim != 2 or not len(x) or not np.isfinite(x).all() or
        edges.ndim != 1 or len(edges) < 2 or not np.issubdtype(edges.dtype, np.integer) or
        edges[0] != 0 or edges[-1] != len(x) or (np.diff(edges) <= 0).any() or
        q.shape != (len(x),) or not np.isfinite(q).all() or (q <= 0).any()):
        raise ValueError('Invalid features, grouped support or positive prior')
    return x, edges, q


def log_probabilities(utility, edges):
    maxima = np.maximum.reduceat(utility, edges[:-1])
    repeated = np.diff(edges)
    mass = np.add.reduceat(np.exp(utility-np.repeat(maxima, repeated)), edges[:-1])
    return utility-np.repeat(maxima+np.log(mass), repeated)


def entropy_projection(utility, edges, q, strength):
    """Scenario-only minimizer KL(p||softmax(utility))+strength*KL(p||q)."""
    utility = np.asarray(utility, float)
    _, edges, q = validate_layout(utility[:, None], edges, q)
    if not np.isfinite(strength) or strength < 0:
        raise ValueError('Entropy strength must be finite and nonnegative')
    logq = log_probabilities(np.log(q), edges)
    scaled = utility/(1+strength)+(strength/(1+strength))*logq
    return np.exp(log_probabilities(scaled, edges))


def loss(theta, x, edges, y, q, strength, ridge):
    """Independent BL plus fractional CE and KL, conditional on observed positives."""
    d = x.shape[1]
    xb = np.column_stack([np.ones(len(x)), x])
    a, b = theta[:d+1], theta[d+1:]
    binary = (y > 0).astype(float)
    logits = xb@a
    value = np.mean(np.logaddexp(0, logits)-binary*logits)
    ga = xb.T@(expit(logits)-binary)/len(x)
    gb = np.zeros(d); groups = 0
    fraction_value = 0.
    for left, right in zip(edges[:-1], edges[1:]):
        positive = np.arange(left, right)[y[left:right] > 0]
        if not len(positive):
            continue
        xx, yy, qq = x[positive], y[positive], q[positive]
        ee = np.array([0, len(positive)])
        lp = log_probabilities(xx@b, ee)
        p = np.exp(lp); target = yy/yy.sum(); logq = log_probabilities(np.log(qq), ee)
        kl = np.dot(p, lp-logq)
        fraction_value += -np.dot(target, lp)+strength*kl
        gb += xx.T@(p-target+strength*p*(lp-logq-kl))
        groups += 1
    if groups:
        value += fraction_value/groups
        gb /= groups
    return float(value+ridge*np.dot(theta, theta)/2), np.r_[ga, gb]+ridge*theta


def fit(x, edges, y, q, *, observation_status, evidence_ref, route_totals,
        expected_sizes, strength=1., ridge=0.01):
    """Caller supplies externally verified completeness, support and route balance."""
    x, edges, q = validate_layout(x, edges, q)
    y = np.asarray(y, float)
    status, evidence = np.asarray(observation_status), np.asarray(evidence_ref)
    totals, expected = np.asarray(route_totals, float), np.asarray(expected_sizes)
    if (y.shape != (len(x),) or not np.isfinite(y).all() or (y < 0).any() or
        (y != np.floor(y)).any() or status.shape != y.shape or evidence.shape != y.shape or
        not (status == 'observed_complete').all() or
        any(not isinstance(v, str) or not v.strip() for v in evidence.tolist()) or
        totals.shape != (len(edges)-1,) or not np.isfinite(totals).all() or
        not np.array_equal(expected, np.diff(edges)) or
        not np.array_equal(np.add.reduceat(y, edges[:-1]), totals)):
        raise ValueError('BL–FS needs complete real stop observations, provenance, support and route balance')
    if not np.isfinite(strength) or strength < 0 or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('Invalid regularization')
    candidates = []
    args = (x, edges, y, q, strength, ridge)
    for initial in [0., -0.1, 0.1]:
        result = minimize(loss, np.full(2*x.shape[1]+1, initial), args=args,
                          jac=True, method='BFGS', options={'gtol': 1e-7, 'maxiter': 1000})
        if np.isfinite(result.fun) and abs(result.jac).max() < 1e-5:
            candidates.append(result)
    if not candidates:
        raise RuntimeError('BL–FS optimization did not converge')
    best = min(candidates, key=lambda v: v.fun)
    return dict(theta=best.x, strength=strength, ridge=ridge, objective=float(best.fun),
                gradient_max=float(abs(best.jac).max()), status='fitted_on_supplied_stop_observations')


def predict(model, x, edges, route_totals):
    x, edges, _ = validate_layout(x, edges, np.ones(len(x)))
    theta = np.asarray(model['theta'], float); d = x.shape[1]
    totals = np.asarray(route_totals, float)
    if (theta.shape != (2*d+1,) or not np.isfinite(theta).all() or
        totals.shape != (len(edges)-1,) or not np.isfinite(totals).all() or (totals < 0).any()):
        raise ValueError('Invalid fitted coefficients or route forecasts')
    activity = expit(np.column_stack([np.ones(len(x)), x])@theta[:d+1])
    shares = np.zeros(len(x)); unassigned = totals.copy()
    for g, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        active = np.arange(left, right)[activity[left:right] >= 0.5]
        if len(active):
            shares[active] = np.exp(log_probabilities(x[active]@theta[d+1:], np.array([0, len(active)])))
            unassigned[g] = 0.
    return dict(nonzero_probability=activity, share=shares,
                boardings=shares*np.repeat(totals, np.diff(edges)), unassigned=unassigned)
