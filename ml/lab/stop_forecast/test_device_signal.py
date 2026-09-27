import numpy as np
from device_signal import density, score

x = np.zeros(360); x[100:103] = [10,20,10]
assert np.isclose(density(x,2).sum(),1) and density(x,2).min()>0
s = score(x,x)
assert s['fast_log_score']>s['slow_log_score']>s['uniform_log_score']
assert s['fast_log_score']>s['mean_placebo_log_score']
uniform = np.ones(360)
np.testing.assert_allclose(density(uniform,2), 1/360)
s = score(x,uniform)
assert abs(s['fast_log_score']-s['uniform_log_score'])<1e-10
print('Device density checks passed')
