"""Conditional time-gap alignment; no Moscow truth or calibrated stop confidence."""
import numpy as np
from scipy.special import logsumexp


def smooth(log_kernels,initial):
    kernels=np.asarray(log_kernels,float);p=np.asarray(initial,float)
    if kernels.ndim!=3 or kernels.shape[1:]!=(len(p),len(p)) or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1):
        raise ValueError('Invalid chain shape/prior')
    if np.isnan(kernels).any() or np.isposinf(kernels).any():
        raise ValueError('Invalid log likelihood')
    with np.errstate(divide='ignore'):
        alpha=[np.log(p)]
    for kernel in kernels:
        alpha.append(logsumexp(alpha[-1][:,None]+kernel,axis=0))
    logz=logsumexp(alpha[-1])
    if not np.isfinite(logz):
        raise ValueError('No feasible chain')
    beta=np.zeros(len(p));posterior=[]
    for t in range(len(alpha)-1,-1,-1):
        state=alpha[t]+beta;posterior.append(np.exp(state-logsumexp(state)))
        if t:
            beta=logsumexp(kernels[t-1]+beta[None,:],axis=1)
    return np.asarray(posterior[::-1]),float(logz)


def align(times,lengths,turns,speed,dwell):
    times=np.asarray(times,float);lengths=np.asarray(lengths,float);turns=np.asarray(turns,bool)
    if len(times)<2 or not np.isfinite(times).all() or (np.diff(times)<=0).any():
        raise ValueError('Expected increasing episode times')
    if lengths.ndim!=1 or len(lengths)<9 or turns.shape!=lengths.shape or not np.isfinite(lengths).all() or (lengths<0).any() or speed<=0 or dwell<0:
        raise ValueError('Invalid graph or timing assumptions')
    n=len(lengths);step=np.arange(9)
    prior=.35*.65**step;prior/=prior.sum()
    means=np.zeros((n,9));means[:,0]=20
    for start in range(n):
        for k in step[1:]:
            edges=(start+np.arange(k))%n
            means[start,k]=max(1,lengths[edges].sum()/(speed/3.6)+dwell*k+60*turns[edges].sum())
    kernels=np.full((len(times)-1,n,n),-np.inf)
    for t,elapsed in enumerate(np.diff(times)):
        mu=np.log(means)-.5*.5**2
        values=-np.log(elapsed*.5*np.sqrt(2*np.pi))-.5*((np.log(elapsed)-mu)/.5)**2+np.log(prior)
        for k in step:
            kernels[t,np.arange(n),(np.arange(n)+k)%n]=values[:,k]
    return smooth(kernels,np.ones(n)/n)


def episodes(high,reference,held):
    high=np.asarray(high,float);reference=np.asarray(reference);held=np.asarray(held)
    if high.shape!=(360,) or reference.shape!=high.shape or held.shape!=high.shape or not np.isfinite(high).all() or ((high<0)|(high>1)).any():
        raise ValueError('Invalid hourly arrays')
    if not np.isfinite(reference).all() or not np.isfinite(held).all() or (reference<0).any() or (held<0).any():
        raise ValueError('Invalid counts')
    mask=high>=.5
    starts=np.flatnonzero(mask & ~np.r_[False,mask[:-1]])
    ends=np.flatnonzero(mask & ~np.r_[mask[1:],False])+1
    result=[]
    for start,end in zip(starts,ends):
        weights=reference[start:end];centres=(np.arange(start,end)+.5)*10
        result.append(dict(first_bin=int(start),end_bin_exclusive=int(end),
            centre_seconds=float(np.average(centres,weights=weights) if weights.sum() else centres.mean()),
            censored=bool(start==0 or end==360),reference_events=int(weights.sum()),
            held_events=int(held[start:end].sum()),mean_high_probability=float(high[start:end].mean())))
    return result
