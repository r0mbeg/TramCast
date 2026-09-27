"""Shared duration calibration and causal, variable-tempo geographic HMM."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import gammaln, xlogy

from activity_model import parameters
from duration_model import duration_transition

WINDOWS = [(m * 6, (m + 2) * 6) for m in (60, 80, 100, 120, 140)] + [(960, 1080)]


def observation_mask():
    mask = np.ones(1080, bool)
    for start, end in WINDOWS:
        mask[start:end] = False
    return mask


def activity_chain(theta):
    duration, factors, _ = parameters(theta)
    transition, survival = duration_transition([[0, 1], [1, 0]], duration)
    return transition, (survival / survival.sum()).ravel(), np.repeat(factors, duration.shape[1])


def geographic_chain(lengths, turns, boarding, theta, persistence, *, fixed=False):
    length = np.asarray(lengths, float)
    turn, board = np.asarray(turns, bool), np.asarray(boarding, bool)
    if (length.ndim != 1 or not len(length) or turn.shape != length.shape or board.shape != length.shape
            or not np.isfinite(length).all() or (length < 0).any() or length.sum() <= 0
            or not board.any() or not np.isfinite(persistence) or not 0 <= persistence <= 1):
        raise ValueError('Invalid geographic chain')
    duration, _, _ = parameters(theta)
    low, high = 10 * (duration @ np.arange(1, duration.shape[1] + 1))
    tempo = np.array([1.]) if fixed else np.array([.67, 1., 1.5])
    denominator = len(length) * low - 60 * turn.sum()
    if denominator <= 0:
        raise ValueError('Duration incompatible with turnaround prior')
    speed = length.sum() * np.mean(1 / tempo) / denominator
    travel = np.maximum(20., length[:, None] / (speed * tempo) + 60 * turn[:, None])
    means = np.stack((np.full_like(travel, max(10., high)), travel / 2, travel / 2), axis=2) / 10
    n, s, _ = means.shape
    q = persistence * np.eye(s) + (1 - persistence) * np.ones((s, s)) / s
    ids = np.arange(n * s * 3).reshape(n, s, 3)
    rows, cols, weights = [], [], []
    for i in range(n):
        for v in range(s):
            for phase in range(3):
                source = ids[i, v, phase]
                leave = 1 / means[i, v, phase]
                rows.append(source); cols.append(source); weights.append(1 - leave)
                if phase < 2:
                    rows.append(source); cols.append(ids[i, v, phase + 1]); weights.append(leave)
                else:
                    for dest in range(s):
                        rows.append(source); cols.append(ids[(i + 1) % n, dest, 0]); weights.append(leave * q[v, dest])
    transition = csr_matrix((weights, (rows, cols)), shape=(ids.size, ids.size))
    initial = (means / means.sum()).ravel()
    factors = np.ones_like(means)
    factors[board, :, 0] = np.exp(theta[0])
    factors = factors.ravel()
    factors /= initial @ factors
    return (transition, initial, factors), dict(effective_base_speed_kmh=float(speed * 3.6),
        registration_mean_seconds=float(high), travel_target_mean_seconds=float(low),
        tempo_levels=tempo.tolist(), states=int(ids.size), persistence=float(persistence))


def filter_intensity(counts, chain, observed, scale):
    """Forward only; hidden bins never enter emissions or the supplied scale."""
    y, mask = np.asarray(counts, float), np.asarray(observed, bool)
    transition, initial, factors = chain
    if (y.ndim != 1 or not len(y) or mask.shape != y.shape or not np.isfinite(y[mask]).all()
            or (y[mask] < 0).any() or (y[mask] != np.floor(y[mask])).any()
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError('Invalid observations or scale')
    rate = scale * factors
    current = initial.copy()
    transpose = transition.T.tocsr()
    intensity = np.empty(len(y))
    logz = 0.
    for t in range(len(y)):
        if t:
            current = transpose @ current
        if mask[t]:
            loge = xlogy(y[t], rate) - rate - gammaln(y[t] + 1)
            maximum = loge.max()
            current *= np.exp(loge - maximum)
            norm = current.sum()
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError('Impossible observation')
            current /= norm
            logz += np.log(norm) + maximum
        intensity[t] = current @ factors
    return intensity, current, float(logz)


def predict(reference, chain):
    reference = np.asarray(reference, float)
    if reference.shape != (1080,):
        raise ValueError('Expected a three-hour sequence')
    intensity, _, _ = filter_intensity(reference, chain, observation_mask(), reference[:360].mean())
    result = []
    for start, end in WINDOWS:
        value = intensity[start:end]
        result.append(.9 * value / value.sum() + .1 / len(value))
    return result


def scores(counts, densities):
    counts = np.asarray(counts)
    if (counts.shape != (1080,) or not np.isfinite(counts).all() or (counts < 0).any()
            or (counts != np.floor(counts)).any() or len(densities) != len(WINDOWS)):
        raise ValueError('Invalid scoring counts')
    short_count = short_score = 0
    for index, ((start, end), p) in enumerate(zip(WINDOWS, densities)):
        if len(p) != end - start or not np.isfinite(p).all() or (p <= 0).any() or not np.isclose(p.sum(), 1):
            raise ValueError('Invalid predictive probabilities')
        count, score = int(counts[start:end].sum()), float(counts[start:end] @ np.log(p))
        if index < 5:
            short_count += count
            short_score += score
        else:
            return dict(short_count=short_count, short_score=short_score, tail_count=count, tail_score=score)
    raise ValueError('Missing forecast windows')
