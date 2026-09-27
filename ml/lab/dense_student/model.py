"""CPU-only student with public external snapshots embedded in its metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from ml.lab.cpu_student.model import features as basic_features, read_history
from ml.lab.cpu_forecast.model import publish

CATS=['route','weekday','route_hour_key']

def workday(dates,calendar):
    result=dates.dt.dayofweek.lt(5)
    result.loc[dates.isin(pd.to_datetime(calendar['holidays']+list(calendar['transfers'].values())))]=False
    result.loc[dates.isin(pd.to_datetime(calendar['working_weekends']))]=True
    return result

def features(history,start,meta):
    keys,x=basic_features(history,start,meta['calendar'],meta['movement'])
    begin=pd.Timestamp(start)
    x['route_hour_key']=x.route+'_'+x.hour.astype(str)
    phase=2*np.pi*(keys.date.dt.dayofyear-1)/365
    x['annual_sin']=np.sin(phase);x['annual_cos']=np.cos(phase)
    x['hour_sin']=np.sin(2*np.pi*x.hour/24);x['hour_cos']=np.cos(2*np.pi*x.hour/24)
    if meta.get('teacher_context_feature',False):
        # Teacher calibration changes only when another monthly 61-day window completes.
        x['completed_teacher_windows']=len(pd.date_range('2025-01-31',begin-pd.Timedelta(days=62),freq='ME'))
    past=history.loc[history.date.lt(begin)&history.working_events_observed&history.route.ne(5)&~history.hour.between(1,4)].copy()
    past['daytype']=np.where(workday(past.date,meta['calendar']),0,np.where(past.date.dt.dayofweek.eq(5),1,2))
    future_type=np.where(x.is_workday.eq(1),0,np.where(keys.date.dt.dayofweek.eq(5),1,2))
    lookup=pd.MultiIndex.from_arrays([keys.route,future_type,keys.hour],names=['route','daytype','hour'])
    for days in [0,14,56,112]:
        pool=past if days==0 else past.loc[past.date.ge(begin-pd.Timedelta(days=days))]
        stats=pool.groupby(['route','daytype','hour']).boardings
        x[f'cal_mean_{days}']=stats.mean().reindex(lookup).to_numpy()
        x[f'cal_median_{days}']=stats.median().reindex(lookup).to_numpy()
    weather=pd.DataFrame(meta['weather']);weather['date']=pd.to_datetime(weather.date)
    weather=weather.set_index('date')
    for col in ['temperature_2m_mean','precipitation_sum','daylight_duration']:
        x[col]=weather[col].reindex(keys.date).to_numpy()
        earlier=weather.loc[(weather.index<begin)&(weather.index>=begin-pd.Timedelta(days=14)),col].mean()
        x[f'origin_{col}']=earlier
    for kind in ['short','summer']:
        flag=pd.Series(False,index=keys.index)
        for interval in meta['school']['intervals']:
            source=next(s for s in meta['school']['sources'] if s['id']==interval['source'])
            if interval['kind']==kind and source.get('known_at') and pd.Timestamp(source['known_at'])<begin:
                flag |= keys.date.between(interval['start'],interval['end'])
        x[f'school_{kind}']=flag.astype(float)
        x.loc[~keys.date.between(*meta['school']['coverage']),f'school_{kind}']=np.nan
    return keys,x

def reference(x,meta):
    return x.cal_mean_0.fillna(x.profile_all).fillna(x.route_hour).fillna(x.route.map(meta['route_scales'])).clip(lower=1).to_numpy()

def matrix(x,kind):
    x=x.copy()
    if kind=='lightgbm':
        for col in CATS:
            if col=='route': values=[str(r) for r in [1,5,7,11,12,17,25,26,28,50]]
            elif col=='weekday':values=[str(i) for i in range(7)]
            else:values=[f'{r}_{h}' for r in [1,5,7,11,12,17,25,26,28,50] for h in range(24)]
            if not x[col].isin(values).all():raise ValueError('Unknown category')
            x[col]=pd.Categorical(x[col],categories=values)
    return x

def predict(model,meta,history,start):
    keys,x=features(history,start,meta)
    if x.columns.tolist()!=meta['features']:raise ValueError('Feature contract mismatch')
    base=reference(x,meta)
    options={'thread_count':2} if meta['kind']=='catboost' else {'num_threads':2}
    raw=model.predict(matrix(x,meta['kind']),**options)
    result=publish(keys,x,base*np.asarray(raw))
    past=history.loc[history.date.lt(pd.Timestamp(start))&history.working_events_observed&history.route.ne(5)]
    warnings=['retrospective_external_weather_scenario']
    if pd.Timestamp(start)<=pd.Timestamp(meta['fit_cutoff']):warnings.append('retrospective_weights_contain_later_history')
    if past.empty:warnings.append('no_history_unvalidated_cold_start')
    elif (pd.Timestamp(start)-past.date.max()).days>1:warnings.append('stale_history')
    if result.date.max().year!=2025:warnings.append('2026_extrapolation_external_coverage_missing')
    if start<'2025-05-01':warnings.append('early_year_student_quality_unvalidated')
    return result,dict(model_version=meta['model_version'],rows=len(result),warnings=warnings,
        latest_history_used=None if past.empty else str(past.date.max().date()),fit_cutoff=meta['fit_cutoff'],
        teacher_latest_origin=meta.get('teacher_latest_origin'),gpu_required=False)

def load_bundle(folder):
    folder=Path(folder);meta=json.loads((folder/'metadata.json').read_text());path=folder/meta['model_file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['model_sha256']:raise ValueError('Model checksum mismatch')
    if meta['kind']=='catboost':
        model=CatBoostRegressor(thread_count=2);model.load_model(str(path))
    else:
        import lightgbm as lgb
        model=lgb.Booster(model_file=str(path))
    return model,meta

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--history',type=Path,required=True);p.add_argument('--start',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    tick=time.perf_counter();model,meta=load_bundle(a.bundle)
    pred,info=predict(model,meta,read_history(a.history),a.start)
    a.output.mkdir(parents=True,exist_ok=False)
    pred.to_csv(a.output/'submission.csv',sep=';',index=False,date_format='%Y-%m-%d')
    info['seconds']=time.perf_counter()-tick
    (a.output/'forecast.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
