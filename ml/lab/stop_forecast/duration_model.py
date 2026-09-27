"""Finite explicit-duration Poisson model, conditional on supplied parameters.

An inference kernel, not a fitted Moscow stop model. Missing observations are
masked explicitly; the unanchored initial age is an equilibrium-age assumption.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import gammaln, xlogy


def duration_transition(next_state, duration):
    a, f = np.asarray(next_state, float), np.asarray(duration, float)
    if f.ndim != 2 or not f.size or a.shape != (len(f), len(f)):
        raise ValueError('Invalid state/duration shapes')
    if not np.isfinite(a).all() or not np.isfinite(f).all() or (a < 0).any() or (f < 0).any():
        raise ValueError('Invalid probabilities')
    if not np.allclose(a.sum(1), 1) or not np.allclose(f.sum(1), 1) or np.diag(a).any():
        raise ValueError('Normalized probabilities and no segment self-transition required')
    survival = np.cumsum(f[:, ::-1], axis=1)[:, ::-1]
    k, d = f.shape
    rows, cols, values = [], [], []
    for state in range(k):
        for age in range(d):
            source = state*d+age
            if survival[state, age] == 0:
                rows.append(source); cols.append(source); values.append(1.)
                continue  # unreachable padding; initial mass must be zero
            hazard = f[state, age]/survival[state, age]
            if age+1 < d:
                rows.append(source); cols.append(source+1); values.append(1-hazard)
            for destination in np.flatnonzero(a[state]):
                rows.append(source); cols.append(destination*d); values.append(hazard*a[state, destination])
    return csr_matrix((values, (rows, cols)), shape=(k*d, k*d)), survival


def infer(counts, rates, next_state, duration, initial_state, *, observed=None, start_at_entry=False):
    """Return smoothed state probabilities and marginal log likelihood.

    counts: time×device integer counts; rates: state×device expectations per bin.
    initial_state is explicit: its choice is not inferred from a first transaction.
    start_at_entry=True is allowed only with an external boundary assumption.
    """
    y, rate, prior = np.asarray(counts, float), np.asarray(rates, float), np.asarray(initial_state, float)
    transition, survival = duration_transition(next_state, duration)
    k, d = survival.shape
    if y.ndim != 2 or not len(y) or rate.shape != (k, y.shape[1]) or prior.shape != (k,):
        raise ValueError('Invalid observation/rate/prior shapes')
    mask = np.ones(y.shape, bool) if observed is None else np.asarray(observed, bool)
    if mask.shape != y.shape or not np.isfinite(y[mask]).all() or (y[mask] < 0).any() or (y[mask] != np.floor(y[mask])).any():
        raise ValueError('Observed counts must be finite nonnegative integers')
    if not np.isfinite(rate).all() or (rate < 0).any() or not np.isfinite(prior).all() or (prior < 0).any() or not np.isclose(prior.sum(), 1):
        raise ValueError('Invalid rates or initial probabilities')
    y = np.where(mask, y, 0)
    log_emission = np.where(mask[:, None, :],
        xlogy(y[:, None, :], rate[None, :, :])-rate[None, :, :]-gammaln(y[:, None, :]+1), 0).sum(2)
    maxima = log_emission.max(1)
    if not np.isfinite(maxima).all():
        raise ValueError('Observations impossible under supplied rates')
    emission = np.repeat(np.exp(log_emission-maxima[:, None]), d, axis=1)
    age = survival/survival.sum(1, keepdims=True)
    if start_at_entry:
        age = np.zeros_like(age); age[:, 0] = 1
    current = (prior[:, None]*age).ravel()
    forward = np.empty_like(emission)
    scales = np.empty(len(y))
    # ponytail: O(T*K*D) storage suffices for pilot sequences; checkpoint for long traces.
    for t in range(len(y)):
        if t:
            current = transition.T@current
        current = current*emission[t]
        scales[t] = current.sum()
        if not np.isfinite(scales[t]) or scales[t] <= 0:
            raise ValueError('No feasible trajectory under supplied parameters')
        current = current/scales[t]
        forward[t] = current
    posterior = np.empty((len(y), k))
    backward = np.ones(k*d)
    for t in range(len(y)-1, -1, -1):
        joint = forward[t]*backward
        posterior[t] = joint.reshape(k, d).sum(1)/joint.sum()
        if t:
            backward = transition@(emission[t]*backward)/scales[t]
    return posterior, float((np.log(scales)+maxima).sum())
