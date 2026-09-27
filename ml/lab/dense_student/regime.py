"""030 parent repair: fill unseen seasonal weekday groups from pre-origin history."""
import numpy as np
import pandas as pd
from experiments.portfolio_structure import tagged, trimmed
from experiments.portfolio_experiment import forecast_keys, complete_raw
from pipeline import KEYS


def regime_forecast(history,cutoff,end,params):
    train=tagged(history.loc[history.date.le(cutoff)&history.route.ne(5)],cutoff)
    train=train.loc[~train.off]
    future=tagged(forecast_keys(cutoff,end),cutoff)
    group=['route','dow','hour']
    def aggregate(pool):
        pool=pool.sort_values('date').groupby(group).tail(params['occurrences'])
        return pool.groupby(group).boardings.agg('mean' if params['statistic']=='mean' else trimmed)
    fallback=aggregate(train);pieces=[]
    for season,target in future.groupby('summer'):
        same=train.loc[train.summer.eq(season)] if params['regime'] else train
        statistic=aggregate(same).combine_first(fallback)
        result=target.merge(statistic.rename('prediction'),on=group,how='left')
        weekend=statistic.loc[statistic.index.get_level_values('dow').isin([5,6])].groupby(['route','hour']).mean()
        holiday=result[KEYS].merge(weekend.rename('holiday'),on=['route','hour'],how='left').holiday
        result.loc[result.off,'prediction']=.95*holiday[result.off].to_numpy()
        result.loc[result.route.eq(5),'prediction']=0.
        pieces.append(result[KEYS+['prediction']])
    raw=complete_raw(pd.concat(pieces,ignore_index=True),cutoff,end)
    if not np.isfinite(raw.prediction).all():raise ValueError('Insufficient past weekday coverage')
    return raw
