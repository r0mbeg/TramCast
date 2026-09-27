"""P70: replay saved P41 Gaussian residuals, without fitting a model."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.portfolio_bayes_direct import annual_features
from experiments.portfolio_experiment import WINDOWS, FINAL, load_history, write_json
from experiments.portfolio_gp_errors import inputs, ROOT

OUT = ROOT / 'robust_annual'


def summarize(frame):
    w = frame.weight.to_numpy(); r = frame.residual.to_numpy()
    mean = np.average(r, weights=w); v = np.average((r-mean)**2, weights=w)
    tail = np.abs(frame.z.to_numpy()) > 3
    return dict(rows=len(frame), days=int(frame.date.nunique()), weighted_mean=float(mean),
        weighted_sd=float(np.sqrt(v)), kurtosis=float(np.average((r-mean)**4, weights=w)/v**2),
        skewness=float(np.average((r-mean)**3, weights=w)/v**1.5),
        beyond_3_sigma=float(np.average(tail, weights=w)),
        tail_squared_error_fraction=float(np.sum(w[tail]*r[tail]**2)/np.sum(w*r**2)),
        clipped_target_fraction=float(np.average(frame.clipped, weights=w)),
        missing_hour_fraction=float(np.average(frame.missing_fraction, weights=w)))


def run():
    history=load_history('artifacts/hourly_clean.csv'); report={}; hashes={}; contexts={}
    observed=history.loc[history.hour.eq(0)|history.hour.ge(5)].groupby(['route','date']).successful_working_observed.mean()
    for cutoff,end in WINDOWS+[FINAL]:
        path=ROOT/'verified_july'/f'posterior_{cutoff}_1.0_224_route_july_verifiedTrue.json'
        meta=json.loads(path.read_text()); past,_=inputs(history,cutoff,end)
        past=past.loc[past.route.ne(5)&past.base.gt(0)].copy()
        past['weight']=1/past.groupby(['route','date']).base.transform('size')
        x=annual_features(past,True); w=past.weight.to_numpy()
        assert len(past)==meta['rows'] and past.date.nunique()==meta['days']
        assert str(past.date.max().date())==meta['latest_target_date'] and past.date.max()<=pd.Timestamp(cutoff)
        np.testing.assert_allclose(np.average(x,axis=0,weights=w),meta['feature_mean'],rtol=1e-12,atol=1e-12)
        scale=np.sqrt(np.average((x-np.array(meta['feature_mean']))**2,axis=0,weights=w))
        scale[scale<1e-14]=1
        np.testing.assert_allclose(scale,meta['feature_scale'],rtol=1e-11,atol=1e-12)
        mu=((x-np.array(meta['feature_mean']))/np.array(meta['feature_scale']))@np.array(meta['coefficient_mean'])+meta['intercept']
        target=np.log((past.boardings.to_numpy()+100)/(past.base.to_numpy()+100))
        past['clipped']=np.abs(target)>np.log(2)
        past['residual']=np.clip(target,-np.log(2),np.log(2))-mu
        past['z']=past.residual*np.sqrt(meta['noise_precision'])
        past['missing_fraction']=1-observed.reindex(pd.MultiIndex.from_frame(past[['route','date']])).to_numpy()
        past['month']=past.date.dt.month
        tail=past.z.abs().gt(3);loss=past.weight*past.residual**2
        contexts[cutoff]=dict(tail_missing_fraction=float(np.average(past.loc[tail,'missing_fraction'],weights=past.loc[tail,'weight'])),
            other_missing_fraction=float(np.average(past.loc[~tail,'missing_fraction'],weights=past.loc[~tail,'weight'])),
            route_tail_loss_fraction=(loss[tail].groupby(past.route[tail]).sum()/loss.sum()).to_dict(),
            route_all_loss_fraction=(loss.groupby(past.route).sum()/loss.sum()).to_dict())
        past.to_csv(OUT/f'residuals_{cutoff}.csv',sep=';',index=False,date_format='%Y-%m-%d')
        report[cutoff]=dict(summary=summarize(past), by_route={str(k):summarize(g) for k,g in past.groupby('route')},
            by_month={str(k):summarize(g) for k,g in past.groupby('month')},
            noise_sd=float(1/np.sqrt(meta['noise_precision'])), features=x.shape[1],
            target_free_future=True, saved_transform_replay=True)
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    hashes.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),Path('artifacts/hourly_clean.csv'),Path('experiments/portfolio_gp_errors.py'),Path('experiments/portfolio_bayes_direct.py')]
        +list((ROOT/'verified_july/inner').glob('*/daily.csv'))})
    write_json(OUT/'diagnostic.json',dict(results=report,sha256=hashes,
        scope='Known-past residuals of P41 Gaussian annual correction; dependent repeated route-days weighted inversely. No model refit or future labels.',
        decision_gate='Robust trial justified if >1% weighted residuals exceed 3 saved noise SD and they contribute >20% weighted squared residuals. This is a diagnostic hypothesis gate, not a score gate.'))
    write_json(OUT/'tail_context.json',contexts)
    print(json.dumps({k:v['summary'] for k,v in report.items()},indent=2))


if __name__=='__main__':run()
