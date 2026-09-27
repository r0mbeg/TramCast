import numpy as np
from activity_model import batch_likelihood,reconstruct,parameters,PRIOR

rng=np.random.default_rng(31415)
counts=rng.poisson(.8,size=(3,15))
ll=batch_likelihood(counts,PRIOR)
for y,expected in zip(counts,ll):
    density,high,actual=reconstruct(y,PRIOR)
    np.testing.assert_allclose(actual,expected,atol=1e-10)
    assert np.isclose(density.sum(),1) and (density>0).all()
    assert ((high>=0)&(high<=1)).all()
d,r,p=parameters(PRIOR)
np.testing.assert_allclose(d.sum(1),1,atol=1e-12)
assert np.isclose(p@r,1) and r[1]>r[0]
try:
    batch_likelihood([[-1,2]],PRIOR)
except ValueError:
    pass
else:
    raise AssertionError('Negative activity count accepted')
print('Batch HSMM matches exact kernel; probabilities and mean rate checked')
