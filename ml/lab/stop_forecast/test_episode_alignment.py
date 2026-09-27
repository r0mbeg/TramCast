from itertools import product
import numpy as np
from episode_alignment import smooth,align,episodes

rng=np.random.default_rng(20260927)
kernel=rng.uniform(.01,1,(3,3,3));initial=np.array([.2,.5,.3])
p,z=smooth(np.log(kernel),initial)
truth=np.zeros_like(p);total=0
for path in product(range(3),repeat=4):
    weight=initial[path[0]]
    for t in range(3):
        weight*=kernel[t,path[t],path[t+1]]
    total+=weight
    for t,state in enumerate(path):
        truth[t,state]+=weight
np.testing.assert_allclose(p,truth/total,atol=1e-12)
assert abs(z-np.log(total))<1e-12
unanchored,_=align([0,100,230,360],np.full(10,300.),np.zeros(10,bool),15,15)
np.testing.assert_allclose(unanchored,.1,atol=1e-12)
anchored,_=smooth(np.log(kernel),[1,0,0])
np.testing.assert_allclose(anchored[0],[1,0,0],atol=1e-12)
high=np.zeros(360);high[:2]=.9;high[10:13]=.9;high[-1]=.9
e=episodes(high,np.ones(360,int),np.ones(360,int))
assert len(e)==3 and [r['censored'] for r in e]==[True,False,True]
assert sum(r['reference_events'] for r in e)==6 and e[1]['centre_seconds']==115
print('Alignment enumeration, unknown phase, anchor and episode boundaries passed')
