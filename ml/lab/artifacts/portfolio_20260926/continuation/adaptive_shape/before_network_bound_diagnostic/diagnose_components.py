"""Posthoc unavailable-fact diagnostics, never a forecast candidate or selection score."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.portfolio_experiment import WINDOWS,load_history,write_json
from experiments.portfolio_combine import raw_frame
from experiments.portfolio_tabular_volume import CONTROL
from pipeline import KEYS

OUT=Path('artifacts/portfolio_20260926/continuation/adaptive_shape')


def run():
    history=load_history('artifacts/hourly_clean.csv');results=[]
    for cutoff,end in WINDOWS:
        raw=raw_frame(CONTROL/f'raw_{cutoff}.csv',cutoff,end)
        truth=history.loc[history.date.gt(cutoff)&history.date.le(end),KEYS+['boardings']].reset_index(drop=True)
        pd.testing.assert_frame_equal(raw[KEYS],truth[KEYS]);frame=raw.merge(truth,on=KEYS,validate='one_to_one')
        y=frame.boardings.to_numpy();p=frame.prediction.to_numpy()
        route_total=frame.groupby(['route','date']).prediction.transform('sum').to_numpy()
        actual_route_total=frame.groupby(['route','date']).boardings.transform('sum').to_numpy()
        net=frame.groupby('date').prediction.transform('sum').to_numpy();actual_net=frame.groupby('date').boardings.transform('sum').to_numpy()
        shape=np.divide(p,route_total,out=np.zeros_like(p),where=route_total>0)
        oracle_profile=np.divide(y,actual_route_total,out=shape.copy(),where=actual_route_total>0)*route_total
        optimal=frame.copy();targets=[]
        for date,g in frame.groupby('date'):
            shares=g.prediction.to_numpy()/g.prediction.sum();actual=g.boardings.to_numpy();positive=shares>0
            ratios=actual[positive]/shares[positive];weights=shares[positive];order=np.argsort(ratios,kind='stable')
            value=ratios[order][np.searchsorted(np.cumsum(weights[order]),weights.sum()/2)]
            left=weights[ratios<value].sum();right=weights[ratios>value].sum();equal=weights[ratios==value].sum()
            assert abs(left-right)<=equal+1e-12
            assert np.abs(value*shares-actual).sum()<=np.abs(actual.sum()*shares-actual).sum()+1e-7
            optimal.loc[g.index,'prediction']=value*shares;targets.append(dict(date=str(date.date()),optimal=value,actual=float(actual.sum()),forecast=float(g.prediction.sum())))
        variants=dict(control=p,actual_network=p*actual_net/net,actual_route=shape*actual_route_total,
            actual_hour_profile=oracle_profile,optimal_network=optimal.prediction.to_numpy())
        for name,values in variants.items():
            assert np.isfinite(values).all() and np.all(values>=0)
            published=np.floor(values+.5).astype(np.int64)
            assert np.all(published[(frame.route.eq(5)|frame.hour.between(1,4)).to_numpy()]==0)
            results.append(dict(cutoff=cutoff,variant=name,score=float(max(0,1-np.abs(published-y).sum()/y.sum())),
                continuous_score=float(max(0,1-np.abs(values-y).sum()/y.sum()))))
        pd.DataFrame(targets).to_csv(OUT/f'unavailable_network_oracle_{cutoff}.csv',sep=';',index=False)
    table=pd.DataFrame(results);table.to_csv(OUT/'component_oracles.csv',sep=';',index=False)
    write_json(OUT/'component_oracles_metadata.json',dict(scope='Posthoc bottleneck isolation with unavailable future facts; not candidates, training examples or selection data. Actual sum is not an L1 oracle upper bound.',
        checks='Frozen keys/targets, structural zeros, nonnegative half-up integers, continuous weighted-median subgradient optimality.',
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('artifacts/hourly_clean.csv')]+list(CONTROL.glob('raw_*.csv'))}))
    print(table.pivot(index='variant',columns='cutoff',values='score').to_string())


if __name__=='__main__':run()
