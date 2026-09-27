"""Two-regime MAP activity model; regimes are not named stops or vehicle motion."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, xlogy
from scipy.stats import nbinom
from duration_model import infer

PRIOR=np.log([5.,11.,2.])
PRIOR_SD=np.array([1.5,1.,1.])
BOUNDS=list(zip(np.log([1.1,1.,.2]),np.log([100.,59.,17.])))


def parameters(theta):
    theta=np.asarray(theta,float)
    if theta.shape!=(3,) or not np.isfinite(theta).all():
        raise ValueError('Expected three finite log parameters')
    contrast=np.exp(theta[0]);extra_mean=np.exp(theta[1:])
    if contrast<=1 or not np.isfinite(extra_mean).all():
        raise ValueError('Invalid activity contrast/duration')
    probability=2/(2+extra_mean)
    duration=nbinom.pmf(np.arange(120)[None,:],2,probability[:,None])
    duration[:,-1]=nbinom.sf(118,2,probability)
    means=duration@np.arange(1,121)
    occupancy=means/means.sum()
    factors=np.array([1.,contrast])/(occupancy[0]+occupancy[1]*contrast)
    return duration,factors,occupancy


def batch_likelihood(counts,theta):
    y=np.asarray(counts,float)
    if y.ndim!=2 or not y.size or not np.isfinite(y).all() or (y<0).any() or (y!=np.floor(y)).any() or (y.sum(1)<=0).any():
        raise ValueError('Expected nonempty complete integer reference sequences')
    duration,factors,_=parameters(theta)
    survival=np.cumsum(duration[:,::-1],axis=1)[:,::-1]
    hazard=np.divide(duration,survival,out=np.ones_like(duration),where=survival>0)
    rate=y.mean(1)[:,None]*factors
    loge=xlogy(y[:,:,None],rate[:,None,:])-rate[:,None,:]-gammaln(y[:,:,None]+1)
    maxima=loge.max(2)
    emission=np.exp(loge-maxima[:,:,None])
    alpha=np.broadcast_to(survival/survival.sum(),(len(y),)+survival.shape).copy()
    logz=np.zeros(len(y))
    for t in range(y.shape[1]):
        if t:
            exits=(alpha*hazard).sum(2)
            nxt=np.zeros_like(alpha)
            nxt[:,:,1:]=alpha[:,:,:-1]*(1-hazard[:,:-1])
            nxt[:,0,0]=exits[:,1];nxt[:,1,0]=exits[:,0]
            alpha=nxt
        alpha*=emission[:,t,:,None]
        scale=alpha.sum((1,2))
        if (scale<=0).any() or not np.isfinite(scale).all():
            raise ValueError('Impossible activity sequence')
        alpha/=scale[:,None,None]
        logz+=np.log(scale)+maxima[:,t]
    return logz


def objective(theta,counts):
    return float(-batch_likelihood(counts,theta).sum()+.5*np.sum(((theta-PRIOR)/PRIOR_SD)**2))


def fit(counts):
    fits,diagnostics=[],[]
    for start in [PRIOR,np.log([3.,5.,1.5]),np.log([10.,20.,4.])]:
        opt=minimize(objective,start,args=(counts,),method='L-BFGS-B',bounds=BOUNDS,
                     options={'maxiter':150,'ftol':1e-11,'gtol':1e-4,'finite_diff_rel_step':1e-5})
        projected=np.asarray(opt.jac).copy()
        for i,(low,high) in enumerate(BOUNDS):
            if (opt.x[i]<=low+1e-7 and projected[i]>0) or (opt.x[i]>=high-1e-7 and projected[i]<0):
                projected[i]=0
        info=dict(success=bool(opt.success),message=str(opt.message),objective=float(opt.fun),
            iterations=int(opt.nit),projected_gradient_max=float(abs(projected).max()),theta=opt.x.tolist())
        diagnostics.append(info)
        print('activity start',len(diagnostics),info,flush=True)
        if opt.success and np.isfinite(opt.fun):
            fits.append(opt)
    if not fits:
        raise RuntimeError('All activity fits failed')
    best=min(fits,key=lambda result:result.fun)
    duration,factors,occupancy=parameters(best.x)
    return best.x,dict(starts=diagnostics,objective=float(best.fun),theta=best.x.tolist(),
        factors=factors.tolist(),stationary_high_probability=float(occupancy[1]),
        duration_mean_seconds=(10*(duration@np.arange(1,121))).tolist(),
        duration_tail_at_cap=duration[:,-1].tolist(),
        active_bounds=[bool(abs(v-lo)<1e-5 or abs(v-hi)<1e-5) for v,(lo,hi) in zip(best.x,BOUNDS)],
        prior_mean=PRIOR.tolist(),prior_sd=PRIOR_SD.tolist(),bounds=BOUNDS)


def reconstruct(reference,theta):
    reference=np.asarray(reference,float)
    if reference.ndim!=1 or not len(reference) or reference.sum()<=0:
        raise ValueError('Nonempty reference required')
    duration,factors,occupancy=parameters(theta)
    posterior,logz=infer(reference[:,None],reference.mean()*factors[:,None],
                         [[0,1],[1,0]],duration,occupancy)
    intensity=posterior@factors
    density=.9*intensity/intensity.sum()+.1/len(reference)
    return density,posterior[:,1],logz
