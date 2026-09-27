"""Run directly: small independent distance and rejection checks."""
import numpy as np
from build_metro_access import distances

assert distances([55.75], [37.6], [55.75], [37.6])[0, 0] == 0
assert np.isclose(distances([0], [0], [0], [1])[0, 0], 111195.08, atol=.01)
a = distances([55, 56], [37, 38], [55, 56], [37, 38])
assert np.allclose(a, a.T) and np.allclose(np.diag(a), 0)
for lat in ([float('nan')], [91]):
    try:
        distances(lat, [37], [55], [37])
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid coordinate accepted')
print('metro access checks passed')
