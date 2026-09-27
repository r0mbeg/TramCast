"""Recalculate frozen forecasts; run from ml with PYTHONPATH=. python .../audit.py."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from archive_submission import archive
from pipeline import KEYS, full_grid, metrics, postprocess, predict

OUT = Path('artifacts/selection_audit_20260927')
PORT = Path('artifacts/portfolio_20260926')
CONT = PORT/'continuation'
OLD = ['2025-04-30', '2025-06-30', '2025-07-31', '2025-08-31']
EXTRA = ['2025-05-31', '2025-08-14']
MAIN = OLD + EXTRA
SOURCES = {'031':CONT/'adaptive_shape/study/adaptive_shape/selected',
    '030':CONT/'tabular_shape/study/tabular_shape/selected',
    '029':CONT/'school_fraction/study/school_fraction/selected',
    '025':CONT/'bayes_shape_timesfm_blend/bayes_shape_timesfm_blend/selected',
    'C0':PORT/'cpu/mean_all', 'C1':PORT/'gpu/daily'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(name, rows):
    result = pd.DataFrame(rows)
    result.to_csv(OUT/f'{name}.csv', sep=';', index=False)
    return result


def raw(method, cutoff):
    end = str((pd.Timestamp(cutoff)+pd.Timedelta(days=61)).date())
    path = (SOURCES[method] if cutoff in OLD else OUT/'new'/method)/f'raw_{cutoff}.csv'
    if not path.exists():
        return None
    frame = pd.read_csv(path, sep=';', parse_dates=['date'], float_precision='round_trip')
    expected = full_grid(pd.Timestamp(cutoff)+pd.Timedelta(days=1),end).to_frame(index=False)
    expected['date'] = pd.to_datetime(expected.date)
    pd.testing.assert_frame_equal(frame[KEYS],expected)
    assert np.isfinite(frame.prediction).all() and frame.prediction.ge(0).all()
    assert frame.loc[frame.route.eq(5)|frame.hour.between(1,4),'prediction'].eq(0).all()
    return frame


def winner(scores):
    stats = scores.groupby('method').wape_score.agg(['mean','min'])
    shortlist = stats.loc[stats['mean'].ge(stats['mean'].max()-.001)]
    return shortlist.sort_values(['min','mean'],ascending=False).index[0]


def regime(frame):
    d = frame.date; weekend = d.dt.dayofweek.ge(5)
    changed = (frame.route.isin([7,50]) & d.between('2025-07-10','2025-08-10'))
    changed |= frame.route.eq(7) & weekend & d.between('2025-08-16','2025-09-05')
    changed |= frame.route.isin([7,50]) & weekend & d.between('2025-09-06','2025-11-14')
    changed |= frame.route.eq(17) & weekend & d.between('2025-04-05','2025-04-30')
    transition = d.between('2025-05-18','2025-06-14')|d.between('2025-08-18','2025-09-14')
    return np.where(changed,'movement',np.where(transition,'transition','stable'))


def run():
    started = time.monotonic()
    history = pd.read_csv('artifacts/hourly_clean.csv',sep=';',parse_dates=['date'])
    assert sha('artifacts/hourly_clean.csv') == '7031c686c149fcb552711df49d82bab23540d8c80ffa6f140c6d0444138384ae'
    expected = full_grid().to_frame(index=False); expected['date']=pd.to_datetime(expected.date)
    pd.testing.assert_frame_equal(history[KEYS],expected)
    labels = pd.concat([pd.read_csv(f'../dataset/labels/labels_day_{n}.csv',sep=';',parse_dates=['date']) for n in ['train','test']])
    joined = history.merge(labels,on=KEYS,how='left',validate='one_to_one',suffixes=('','_labels'))
    assert joined.raw_boardings.eq(joined.boardings_labels.fillna(0)).all()
    table('data_contract',[dict(rows=len(history),raw_total=int(history.raw_boardings.sum()),
        clean_total=int(history.boardings.sum()),removed=int(history.removed_boardings.sum()),
        missing_active_cells=int((history.route.ne(5)&~history.hour.between(1,4)&~history.working_events_observed).sum()),
        raw_labels_exact=True)])
    rows=[]; sensitivities=[]; slices=[]; contributions=[]; missing=[]; lineage=[]
    all_dates = MAIN+['2025-05-29','2025-06-02','2025-08-12','2025-08-16']
    forecasts={}
    for cutoff in all_dates:
        end = pd.Timestamp(cutoff)+pd.Timedelta(days=61)
        truth = history.loc[history.date.gt(cutoff)&history.date.le(end)].reset_index(drop=True)
        active=truth.route.ne(5)&~truth.hour.between(1,4)
        observed=~active|truth.working_events_observed
        train=history.loc[history.date.le(cutoff)&history.route.ne(5)&~history.hour.between(1,4)&history.working_events_observed].copy()
        train['weekday']=train.date.dt.dayofweek
        profile=train.groupby(['route','weekday','hour']).boardings.median()
        lookup=pd.MultiIndex.from_arrays([truth.route,truth.date.dt.dayofweek,truth.hour])
        estimated=profile.reindex(lookup).to_numpy()
        imputed=truth.boardings.to_numpy(float).copy(); absent=active&~truth.working_events_observed
        assert np.isfinite(estimated[absent]).all()
        imputed[absent]=estimated[absent]
        for method in SOURCES:
            frame=raw(method,cutoff)
            if frame is None:
                missing.append(dict(method=method,cutoff=cutoff)); continue
            published=postprocess(frame)
            pd.testing.assert_frame_equal(published[KEYS],truth[KEYS])
            forecasts[method,cutoff]=published
            values=published.prediction.to_numpy()
            row=dict(method=method,cutoff=cutoff,end=str(end.date()),role='old' if cutoff in OLD else 'additional' if cutoff in EXTRA else 'shift',
                rows=len(published),**metrics(truth.boardings,values))
            rows.append(row)
            for name,y,keep in [('clean',truth.boardings,np.ones(len(truth),bool)),
                                ('raw_labels',truth.raw_boardings,np.ones(len(truth),bool)),
                                ('observed_only',truth.boardings,observed),
                                ('past_median_missing',imputed,np.ones(len(truth),bool))]:
                sensitivities.append(dict(method=method,cutoff=cutoff,scenario=name,rows=int(np.sum(keep)),**metrics(np.asarray(y)[keep],values[keep])))
            masks={'regime':regime(truth),'route':truth.route.astype(str),
                'month':truth.date.dt.strftime('%Y-%m'), 'daytype':np.where(truth.date.dt.dayofweek.ge(5),'weekend','weekday'),
                'horizon':pd.cut((truth.date-pd.Timestamp(cutoff)).dt.days,[0,14,30,61],labels=['1-14','15-30','31-61']).astype(str)}
            for dim,groups in masks.items():
                for group in sorted(set(groups)):
                    keep=np.asarray(groups)==group
                    slices.append(dict(method=method,cutoff=cutoff,dimension=dim,group=group,rows=int(keep.sum()),**metrics(truth.boardings.to_numpy()[keep],values[keep])))
    windows=table('windows',rows);table('missing_forecasts',missing);table('sensitivity',sensitivities)
    breakdown=table('breakdown',slices)
    summary=[]
    for name,dates in [('old_four',OLD),('additional_two',EXTRA),('main_six',MAIN)]:
        for method in SOURCES:
            part=windows.loc[windows.method.eq(method)&windows.cutoff.isin(dates)]
            complete=len(part)==len(dates)
            summary.append(dict(scope=name,method=method,expected_windows=len(dates),available_windows=len(part),complete=complete,
                mean_score=part.wape_score.mean() if complete else None,
                worst_score=part.wape_score.min() if complete else None,
                pooled_score=max(0,1-part.absolute_error.sum()/part.actual_total.sum()) if complete else None))
    table('comparison',summary)
    robustness=[]
    for name,dates in [('old_four',OLD),('main_six',MAIN)]:
        chosen=windows.loc[windows.cutoff.isin(dates)]
        if not chosen.groupby('method').size().eq(len(dates)).all():continue
        for omit in [None]+dates:
            part=chosen if omit is None else chosen.loc[chosen.cutoff.ne(omit)]
            robustness.append(dict(check=name,omitted=omit,winner=winner(part)))
    for offset in [-2,2]:
        shifted=[str((pd.Timestamp(c)+pd.Timedelta(days=offset)).date()) for c in EXTRA]
        for name,dates in [('shifted_extra',shifted),('shifted_main',OLD+shifted)]:
            chosen=windows.loc[windows.cutoff.isin(dates)]
            if len(chosen)==len(dates)*len(SOURCES):robustness.append(dict(check=name,offset=offset,winner=winner(chosen)))
    sens=pd.DataFrame(sensitivities)
    for scenario in sens.scenario.unique():
        for name,dates in [('old_four',OLD),('main_six',MAIN)]:
            part=sens.loc[sens.scenario.eq(scenario)&sens.cutoff.isin(dates)]
            if len(part)==len(dates)*len(SOURCES):robustness.append(dict(check=name,scenario=scenario,winner=winner(part)))
    table('robustness',robustness)
    for cutoff in OLD:
        end=str((pd.Timestamp(cutoff)+pd.Timedelta(days=61)).date())
        truth=history.loc[history.date.gt(cutoff)&history.date.le(end)].reset_index(drop=True)
        for experiment,a,b in [('movement_notice',PORT/'ridge_probe/ridge/selected',PORT/'movement'),
            ('school_proxy',CONT/'route_fraction/study/route_fraction/selected',SOURCES['029'])]:
            pair=[]
            for folder in [a,b]:
                frame=pd.read_csv(folder/f'raw_{cutoff}.csv',sep=';',parse_dates=['date'],float_precision='round_trip')
                pd.testing.assert_frame_equal(frame[KEYS],truth[KEYS]); pair.append(postprocess(frame))
            for group in ['all','movement','transition','stable']:
                keep=np.ones(len(truth),bool) if group=='all' else regime(truth)==group
                left=metrics(truth.boardings.to_numpy()[keep],pair[0].prediction.to_numpy()[keep])
                right=metrics(truth.boardings.to_numpy()[keep],pair[1].prediction.to_numpy()[keep])
                contributions.append(dict(experiment=experiment,cutoff=cutoff,group=group,
                    without_score=left['wape_score'],with_score=right['wape_score'],
                    delta=right['wape_score']-left['wape_score'] if left['wape_score'] is not None else None))
    table('external_pairs',contributions)
    for method,folder in SOURCES.items():
        choices=list(folder.parent.glob('selection.json'))
        for path in choices:
            info=json.loads(path.read_text());latest=max(pd.Timestamp(e) for _,e in info.get('selection_windows',[]))
            for cutoff in OLD:
                lineage.append(dict(method=method,cutoff=cutoff,selection_file=str(path),
                    latest_selection_target=str(latest.date()),selection_completed_at_cutoff=latest<=pd.Timestamp(cutoff)))
    table('selection_timing',lineage)
    exports={}
    for method,pattern,attempt in [('031','031_*.csv','recommended_031'),('025','025_*.csv','alternative_025'),('C1','002_*.csv','alternative_002')]:
        source=next(Path('submissions').glob(pattern));target=OUT/'exports'/f'{attempt}.csv'
        if not target.exists():target=archive(source,attempt,directory=OUT/'exports')
        assert sha(source)==sha(target)
        exports[method]=dict(path=str(target),source=str(source),sha256=sha(target),rows=14640)
    old=json.loads((OUT/'protected_manifest.json').read_text())['files']
    changed=[path for path,digest in old.items() if not Path(path).is_file() or sha(path)!=digest]
    assert not changed,changed
    write=dict(seconds=time.monotonic()-started,protected_files=len(old),protected_changed=changed,
        exports=exports,main_complete=len(windows.loc[windows.cutoff.isin(MAIN)])==36,
        windows_evaluated=len(windows),new_model_trials=0)
    (OUT/'verification.json').write_text(json.dumps(write,indent=2))
    print(pd.DataFrame(summary).to_string(index=False));print('robustness',robustness)


def self_check():
    assert metrics([10,20],[8,22])['wape_score']==1-4/30
    frame=pd.DataFrame({'route':[1,1,1,5,1],'date':['2025-11-01']*5,'hour':[0,5,6,0,1],
        'prediction':[.5,1.5,-.5,99.,99.]})
    assert postprocess(frame).prediction.tolist()==[1,2,0,0,0]
    x=pd.DataFrame({'method':['a','a','b','b'],'wape_score':[.9,.88,.8999,.8802]})
    assert winner(x)=='b'
    altered=frame.copy();altered['date']=pd.to_datetime(altered.date)
    assert set(regime(altered))=={'stable'}


if __name__=='__main__':
    self_check();run()
