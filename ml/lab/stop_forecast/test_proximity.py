"""Proximity join is key-based; missing features never silently become zeros."""
import numpy as np
import pandas as pd
from proximity_experiment import features

stops = pd.DataFrame(dict(route=[1, 1], trip_id=[10, 10], stop_sequence=[1, 2]))
access = stops.assign(nearest_2021_entrance_distance_m=[10, 1000],
                      metro_2021_station_line_candidates_500m=[2, 0])
a, _ = features(stops, access)
b, _ = features(stops, access.iloc[::-1])
np.testing.assert_allclose(a, b)
np.testing.assert_allclose(a.mean(axis=0), 0, atol=1e-12)
assert a[0, 0] < a[1, 0] and a[0, 1] > a[1, 1]
try:
    features(stops, access.iloc[:1])
except ValueError:
    pass
else:
    raise AssertionError('Missing feature accepted')
print('proximity checks passed')
