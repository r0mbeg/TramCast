"""Bounded CPU distillation, predeclared time splits and pseudo-target ablation."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import signal
import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
from ml.lab.cpu_student.model import read_history, grid, rounded
from ml.lab.cpu_student.train import digest, save_json, measure
from ml.lab.cpu_forecast.train import evaluated
from ml.lab.dense_student.model import CATS, features, reference, matrix, predict, load_bundle
from ml.runtime.constants import KEYS

LAB=Path(__file__).resolve().parents[1]
CANDIDATES=[dict(kind=k,alpha=a) for k in ['catboost','lightgbm'] for a in [1.,.75,.5]]+[dict(kind='lightgbm',alpha=0.)]
DEV=['2025-07-01','2025-08-01']
EXTERNAL=LAB/'artifacts/portfolio_20260926/continuation/external'


def context():
    return dict(calendar=json.loads((LAB/'artifacts/calendar_sources.json').read_text()),
        movement=json.loads((LAB.parent/'preparation/movement_calendar.json').read_text()),
        school=json.loads((EXTERNAL/'school_calendar/sources.json').read_text()),
        weather=pd.read_csv(EXTERNAL/'weather_2025.csv',sep=';').to_dict('list'))


def examples(history,cutoff,common,teachers,purge=False):
    edge=pd.Timestamp(cutoff);actual=[];soft=[]
    truth=history.loc[history.date.le(edge)&history.working_events_observed&history.route.ne(5)&~history.hour.between(1,4)]
    scales={**truth.groupby('route').boardings.median().clip(lower=1).rename(index=str).to_dict(),'5':1.}
    starts=pd.date_range('2025-01-29',edge,freq='14D')
    for start in starts:
        keys,x=features(history,str(start.date()),common)
        paired=keys.merge(truth[KEYS+['boardings']],on=KEYS,how='left',validate='one_to_one')
        use=paired.boardings.notna()
        actual.append(x.loc[use].assign(target=paired.loc[use,'boardings'].to_numpy(),
            target_date=keys.loc[use,'date'].to_numpy(),origin=start-pd.Timedelta(days=1)))
    origins=[]
    for path in sorted(teachers.glob('raw_*.csv')):
        origin=pd.Timestamp(path.stem[4:])
        if origin>edge:continue
        info=json.loads(path.with_suffix('.json').read_text())
        if digest(path)!=info['sha256'] or info['history_max']!=str(origin.date()):raise ValueError('Invalid teacher lineage')
        start=str((origin+pd.Timedelta(days=1)).date());keys,x=features(history,start,common)
        raw=pd.read_csv(path,sep=';',parse_dates=['date']).sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(raw[KEYS],keys)
        if not np.isfinite(raw.prediction).all() or raw.prediction.lt(0).any():raise ValueError('Invalid teacher')
        if not raw.loc[keys.route.eq(5)|keys.hour.between(1,4),'prediction'].eq(0).all():
            raise ValueError('Teacher structural zeros changed')
        use=keys.route.ne(5)&~keys.hour.between(1,4)
        if purge:use &= keys.date.le(edge)
        soft.append(x.loc[use].assign(target=raw.loc[use,'prediction'].to_numpy(),
            target_date=keys.loc[use,'date'].to_numpy(),origin=origin))
        origins.append(str(origin.date()))
    actual=pd.concat(actual,ignore_index=True);soft=pd.concat(soft,ignore_index=True)
    assert actual.target_date.max()<=edge and soft.origin.max()<=edge
    if purge:assert soft.target_date.max()<=edge
    audit=dict(fit_cutoff=cutoff,actual_rows=len(actual),teacher_rows=len(soft),teacher_origins=origins,
        actual_latest_target=str(actual.target_date.max().date()),teacher_latest_target=str(soft.target_date.max().date()),
        teacher_targets_after_cutoff=int(soft.target_date.gt(edge).sum()),purged_teacher_targets=purge)
    return actual,soft,dict(common,route_scales=scales,fit_cutoff=cutoff,teacher_latest_origin=max(origins)),audit


def fit(actual,soft,meta,config):
    parts=[];ys=[];weights=[]
    for data,fraction in [(soft,config['alpha']),(actual,1-config['alpha'])]:
        if fraction==0:continue
        x=data.drop(columns=['target','target_date','origin']);base=reference(x,meta)
        repeat=data.groupby(['route','target_date','hour']).target.transform('size').to_numpy()
        weight=(base**(2 if config['kind']=='catboost' else 1))/repeat
        weight=weight/weight.sum()*fraction
        parts.append(x);ys.append(data.target.to_numpy()/base);weights.append(weight)
    x=pd.concat(parts,ignore_index=True);y=np.concatenate(ys);w=np.concatenate(weights);w*=len(w)
    if config['kind']=='catboost':
        model=CatBoostRegressor(depth=7,iterations=800,loss_function='RMSE',learning_rate=.05,
            thread_count=2,random_seed=42,task_type='CPU',allow_writing_files=False,verbose=False)
        model.fit(x,y,cat_features=CATS,sample_weight=w)
    else:
        model=lgb.LGBMRegressor(objective='regression_l1',num_leaves=31,n_estimators=800,
            learning_rate=.05,min_child_samples=40,reg_lambda=1,n_jobs=2,random_state=42,
            deterministic=True,force_col_wise=True,verbosity=-1)
        model.fit(matrix(x,'lightgbm'),y,sample_weight=w,categorical_feature=CATS)
    return model,dict(meta,kind=config['kind'],features=x.columns.tolist(),config=config,model_version='evaluation')


def save_bundle(folder,model,meta):
    folder.mkdir(parents=True,exist_ok=False)
    name='student.cbm' if meta['kind']=='catboost' else 'student.lgb'
    path=folder/name
    if meta['kind']=='catboost':model.save_model(str(path))
    else:model.booster_.save_model(str(path))
    h=digest(path)
    meta=dict(meta,model_file=name,model_sha256=h,model_version='dense-030-cpu-'+h[:12],
        license_note='Built with PriorLabs-TabPFN; distilled outputs of 030, see ml/recipes/tabpfn-030/LICENSE.txt')
    save_json(folder/'metadata.json',meta)
    return meta


def run(out,teacher_dir,phase='all'):
    begun=time.perf_counter();out.mkdir(parents=True,exist_ok=True)
    if phase!='finish' and (out/'selection.json').exists():raise FileExistsError('Completed selection exists')
    if phase=='finish':
        selection=json.loads((out/'selection.json').read_text())
        previous=json.loads((out/'protocol_development.json').read_text())
        if any(digest(Path(p))!=h for p,h in previous['sources_sha256'].items()):
            raise ValueError('Development inputs or source changed')
    expected={str(d.date()) for d in pd.date_range('2025-04-09','2025-10-29',freq='7D')}|{'2025-05-31'}
    available={p.stem[4:] for p in teacher_dir.glob('raw_*.csv')}
    required={d for d in expected if d<='2025-07-31'} if phase=='development' else expected
    if not required<=available or available-expected:raise ValueError(f'Teacher schedule incomplete or changed: missing {required-available}, extra {available-expected}')
    common=context();history=read_history(LAB/'artifacts/hourly_clean.csv')
    inputs=[Path(__file__),Path(__file__).with_name('model.py'),out/'PROTOCOL.md',LAB/'artifacts/hourly_clean.csv',
        LAB/'artifacts/calendar_sources.json',LAB.parent/'preparation/movement_calendar.json',
        EXTERNAL/'weather_2025.csv',EXTERNAL/'school_calendar/sources.json',LAB/'cpu_student/model.py',
        LAB/'cpu_forecast/model.py',LAB/'cpu_student/train.py',LAB/'cpu_forecast/train.py',
        LAB/'artifacts/portfolio_20260926/continuation/tabular_shape/study/tabular_shape/selected/raw_2025-08-31.csv',
        LAB.parent/'bundles/030/forecast.csv',
        *sorted(teacher_dir.glob('*'))]
    if phase=='finish':inputs.append(out/'selection.json')
    hashes={str(p):digest(p) for p in inputs if p.is_file()}
    save_json(out/f'protocol_{phase}.json',dict(candidates=CANDIDATES,development=DEV,test='2025-09-01',
        sources_sha256=hashes,selection='mean observed WAPE on development',gpu_in_student=False,
        versions={n:importlib.metadata.version(n) for n in ['catboost','lightgbm','numpy','pandas']},
        platform=platform.platform(),
        external_weather='retrospective ERA5 scenario, not historical weather forecast',budget_seconds=1800))
    rows=[];audits=[]
    for start in ([] if phase=='finish' else DEV):
        cutoff=str((pd.Timestamp(start)-pd.Timedelta(days=1)).date())
        actual,soft,meta,audit=examples(history,cutoff,common,teacher_dir);audits.append(audit)
        for i,config in enumerate(CANDIDATES):
            tick=time.perf_counter();model,info=fit(actual,soft,meta,config)
            prediction,_=predict(model,info,history,start);paired,metric=evaluated(history,prediction)
            rows.append(dict(start=start,candidate=i,**config,seconds=time.perf_counter()-tick,**metric))
            pd.DataFrame(rows).to_csv(out/'development.csv',sep=';',index=False)
            print('DEV',start,i,config,'WAPE',metric['wape'],'seconds',rows[-1]['seconds'],flush=True)
    if phase!='finish':
        scores=pd.DataFrame(rows).groupby('candidate').wape.mean();selected=int(scores.idxmin());config=CANDIDATES[selected]
        save_json(out/'selection.json',dict(candidate=selected,config=config,mean_wape=float(scores[selected]),scores=scores.to_dict()))
    else:config=selection['config']
    if phase=='development':
        if any(digest(Path(p))!=h for p,h in hashes.items()):raise ValueError('Development sources changed')
        save_json(out/'development_audit.json',audits)
        save_json(out/'development_completed.json',dict(seconds=time.perf_counter()-begun))
        print('SELECTION COMPLETE',config,flush=True)
        return
    checks=[];details=[];fidelity=[]
    teacher_test=pd.read_csv(LAB/'artifacts/portfolio_20260926/continuation/tabular_shape/study/tabular_shape/selected/raw_2025-08-31.csv',sep=';',parse_dates=['date'])
    teacher_test=rounded(teacher_test[KEYS],teacher_test.prediction).sort_values(KEYS).reset_index(drop=True)
    for purge in [False,True]:
        actual,soft,meta,audit=examples(history,'2025-08-31',common,teacher_dir,purge);audits.append(audit)
        model,info=fit(actual,soft,meta,config)
        label='purged_transfer' if purge else 'teacher_assisted'
        saved=save_bundle(out/label,model,info)
        pred,_=predict(model,saved,history,'2025-09-01');paired,metric=evaluated(history,pred)
        pd.testing.assert_frame_equal(pred[KEYS],teacher_test[KEYS])
        fidelity.append(dict(model=label,start='2025-09-01',target='heldout_exact_origin_030',**measure(teacher_test.prediction,pred.prediction)))
        pred.to_csv(out/f'{label}_september_october.csv',sep=';',index=False,date_format='%Y-%m-%d')
        checks.append(dict(model=label,**metric));print('TEST',label,metric,flush=True)
        for col in ['route','hour']:
            for value,group in paired.groupby(col):details.append(dict(model=label,dimension=col,value=int(value),**measure(group.boardings,group.prediction)))
        paired['week']=((paired.date-pd.Timestamp('2025-09-01')).dt.days//7+1)
        for value,group in paired.groupby('week'):details.append(dict(model=label,dimension='horizon_week',value=int(value),**measure(group.boardings,group.prediction)))
    for label,path in [('cpu_catboost',LAB/'artifacts/cpu_forecast_20260927_v2/test_september_october.csv'),
                       ('lightgbm_laplace',LAB/'artifacts/lightgbm_laplace_20260927_v1/test_september_october.csv'),
                       ('gpu_030',LAB/'artifacts/portfolio_20260926/continuation/tabular_shape/study/tabular_shape/selected/raw_2025-08-31.csv')]:
        raw=pd.read_csv(path,sep=';',parse_dates=['date']);pred=rounded(raw[KEYS],raw.prediction)
        _,metric=evaluated(history,pred);checks.append(dict(model=label,**metric))
    pd.DataFrame(checks).to_csv(out/'test_metrics.csv',sep=';',index=False)
    pd.DataFrame(details).to_csv(out/'breakdown.csv',sep=';',index=False)
    actual,soft,meta,audit=examples(history,'2025-10-31',common,teacher_dir);audits.append(audit)
    tick=time.perf_counter();model,info=fit(actual,soft,meta,config);final_seconds=time.perf_counter()-tick
    meta=save_bundle(out/'bundle',model,info)
    tick=time.perf_counter();submission,_=predict(model,meta,history,'2025-11-01');inference=time.perf_counter()-tick
    pd.testing.assert_frame_equal(submission[KEYS],grid('2025-11-01'))
    submission.to_csv(out/'submission.csv',sep=';',index=False,date_format='%Y-%m-%d')
    teacher_final=pd.read_csv(LAB.parent/'bundles/030/forecast.csv',sep=';',parse_dates=['date']).sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(submission[KEYS],teacher_final[KEYS])
    fidelity.append(dict(model='final_refit',start='2025-11-01',target='heldout_exact_origin_030_no_actual_targets',**measure(teacher_final.prediction,submission.prediction)))
    pd.DataFrame(fidelity).to_csv(out/'teacher_fidelity.csv',sep=';',index=False)
    loaded,metadata=load_bundle(out/'bundle');again,_=predict(loaded,metadata,history,'2025-11-01')
    pd.testing.assert_frame_equal(submission,again)
    for start in ['2025-01-01','2025-09-15','2025-12-31']:
        pred,_=predict(loaded,metadata,history,start);pd.testing.assert_frame_equal(pred[KEYS],grid(start))
        assert len(pred)==14640 and pred.prediction.dtype==np.dtype('int64') and pred.prediction.ge(0).all()
        assert pred.loc[pred.route.eq(5)|pred.hour.between(1,4),'prediction'].eq(0).all()
    before,_=predict(loaded,metadata,history,'2025-09-15')
    poisoned=history.copy();future=poisoned.date.ge('2025-09-15')&poisoned.route.ne(5)&~poisoned.hour.between(1,4)
    poisoned.loc[future,'boardings']=987654321;poisoned.loc[future,'working_events_observed']=False
    after,_=predict(loaded,metadata,poisoned,'2025-09-15');pd.testing.assert_frame_equal(before,after)
    if any(digest(Path(p))!=h for p,h in hashes.items()):raise ValueError('Source changed during training')
    save_json(out/'training_audit.json',audits)
    save_json(out/'verification.json',dict(native_reload_exact=True,future_target_and_mask_poison_unchanged=True,
        full_grid=True,rows=len(submission),nonnegative_int64=True,structural_zeros=True,
        year_boundary_starts_checked=True,source_hashes_unchanged=True,
        gpu_modules_loaded=[n for n in ['torch','tabpfn','timesfm'] if n in sys.modules],submission_sha256=digest(out/'submission.csv')))
    prior_seconds=json.loads((out/'development_completed.json').read_text())['seconds'] if phase=='finish' else 0
    save_json(out/'completed.json',dict(seconds=time.perf_counter()-begun+prior_seconds,final_fit_seconds=final_seconds,
        inference_seconds=inference,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
        model_bytes=(out/'bundle'/meta['model_file']).stat().st_size))
    print('COMPLETE',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--teachers',type=Path,required=True)
    p.add_argument('--phase',choices=['all','development','finish'],default='all')
    a=p.parse_args()
    spent=json.loads((a.output/'development_completed.json').read_text())['seconds'] if a.phase=='finish' else 0
    if spent>=1800:raise RuntimeError('CPU budget exhausted')
    signal.alarm(max(1,int(1800-spent)))
    run(a.output.resolve(),a.teachers.resolve(),a.phase)
