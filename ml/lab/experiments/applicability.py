"""M20260927 fixed controls, information ablation and published-hour evaluation."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import pandas as pd

from experiments.chronos_experiment import REVISION
from experiments.portfolio_chronos import gpu_forecast
from experiments.portfolio_experiment import forecast_keys, load_history
from experiments.portfolio_movement import disrupted
from experiments.portfolio_ridge import ridge_forecast
from pipeline import KEYS, metrics, postprocess, predict

ORIGINS = ['2025-05-01', '2025-05-15', '2025-06-01', '2025-06-15', '2025-07-01',
           '2025-07-15', '2025-08-01', '2025-08-15', '2025-09-01']
HORIZONS = [1, 7, 30, 61]
OLD = ['2025-04-30', '2025-05-31', '2025-06-30', '2025-07-31', '2025-08-31']
EVENTS = {'july': '2025-07-09', 'autumn': '2025-09-05'}
PARAMS = dict(correction_days=14, shape_mix=0.5, route_season=False)
WEIGHTS_SHA256 = 'ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    if path.exists():
        raise ValueError(f'Refusing overwrite: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, pd.DataFrame):
        value.to_csv(path, sep=';', index=False, date_format='%Y-%m-%d')
    else:
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def segments(origin, mode):
    start = pd.Timestamp(origin)
    finish = start + pd.Timedelta(days=60)
    for date in ([start] if mode == 'frozen' else pd.date_range(start, finish, freq='7D')):
        yield date - pd.Timedelta(days=1), min(finish, date + pd.Timedelta(days=6)) if mode == 'weekly' else finish


def c0_raw(history, cutoff, end):
    train = history.loc[history.date.le(cutoff)].assign(weekday=lambda x: x.date.dt.dayofweek)
    keys = forecast_keys(cutoff, end).assign(weekday=lambda x: x.date.dt.dayofweek)
    profile = train.groupby(['route', 'weekday', 'hour']).boardings.mean().rename('prediction')
    raw = keys.merge(profile, on=['route', 'weekday', 'hour'], validate='many_to_one')[KEYS + ['prediction']]
    pd.testing.assert_frame_equal(postprocess(raw), predict(history, keys, cutoff, 0, 'mean'))
    return raw


def forecast(history, method, cutoff, end, predictor=None):
    train = history.loc[history.date.le(cutoff)].copy()
    assert train.date.max() <= pd.Timestamp(cutoff)
    return c0_raw(train, cutoff, end) if method == 'C0' else gpu_forecast(train, cutoff, end, 'daily', predictor)


def regime(frame):
    d = frame.date; weekend = d.dt.dayofweek.ge(5)
    changed = frame.route.isin([7, 50]) & d.between('2025-07-10', '2025-08-10')
    changed |= frame.route.eq(7) & weekend & d.between('2025-08-16', '2025-09-05')
    changed |= frame.route.isin([7, 50]) & weekend & d.between('2025-09-06', '2025-11-14')
    changed |= frame.route.eq(17) & weekend & d.between('2025-04-05', '2025-04-30')
    transition = d.between('2025-05-18', '2025-06-14') | d.between('2025-08-18', '2025-09-14')
    return np.where(changed, 'movement', np.where(transition, 'transition', 'stable'))


def measured(frame, level):
    columns = {'hourly': KEYS, 'daily': ['route', 'date'],
               'route_calendar_month': ['route', 'month']}[level]
    grouped = frame.groupby(columns, observed=True)[['boardings', 'prediction']].sum()
    row = metrics(grouped.boardings, grouped.prediction)
    total = row['actual_total']
    return dict(row, n=len(grouped), wape=row['absolute_error'] / total if total else None,
                mae=row['absolute_error'] / len(grouped) if len(grouped) else None,
                predicted_total=float(grouped.prediction.sum()), bias_ratio=row['bias'] / total if total else None)


def evaluate(history, published, spec):
    truth = history.loc[history.date.isin(published.date.unique())]
    joined = truth.merge(published, on=KEYS, validate='one_to_one')
    if len(joined) != len(truth) or len(joined) != len(published):
        raise ValueError('Incomplete evaluation grid')
    joined['month'] = joined.date.dt.strftime('%Y-%m')
    joined['elapsed'] = (joined.date - pd.Timestamp(spec['origin'])).dt.days + 1
    joined['degradation'] = pd.cut(joined.elapsed, [0, 7, 30, 61], labels=['1-7', '8-30', '31-61'])
    joined['regime'] = regime(joined)
    rows, slices = [], []
    for horizon in HORIZONS:
        group = joined.loc[joined.elapsed.le(horizon)]
        for level in ['hourly', 'daily', 'route_calendar_month']:
            rows.append(dict(spec, horizon=horizon, level=level, **measured(group, level)))
        for dimension in ['route', 'month', 'degradation', 'regime']:
            for value, part in group.groupby(dimension, observed=True):
                for level in ['hourly', 'daily', 'route_calendar_month']:
                    slices.append(dict(spec, horizon=horizon, dimension=dimension, value=str(value), level=level,
                        partial_month=part.date.min().day != 1 or part.date.max().day != part.date.max().days_in_month,
                        **measured(part, level)))
    return rows, slices


def checkpoint(history, raw, root, metadata):
    raw = raw.sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(raw[KEYS], forecast_keys(metadata['cutoff'], metadata['end']))
    write(root / 'raw.csv', raw)
    published = postprocess(raw)
    write(root / 'published.csv', published)
    write(root / 'trace.json', dict(metadata, raw_sha256=sha(root / 'raw.csv'),
        published_sha256=sha(root / 'published.csv'), rows=len(raw),
        input_rows=len(history), input_max=str(history.date.max().date()),
        input_sha256=hashlib.sha256(history[KEYS + ['boardings']].to_csv(index=False, date_format='%Y-%m-%d').encode()).hexdigest()))
    return published


def run_gpu(args):
    import torch
    from chronos import Chronos2Pipeline
    if os.environ.get('SLURM_JOB_PARTITION') != 'gpu_devel' or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one gpu_devel GPU')
    if int(os.environ['SLURM_CPUS_PER_TASK']) != 2 or len(os.sched_getaffinity(0)) > 2:
        raise RuntimeError('Expected two bound CPU cores')
    started = time.monotonic(); cpu_started = time.process_time()
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(42); np.random.seed(42)
    root = Path(args.output); history = load_history(args.history)
    files = [Path(args.history), Path(__file__), Path('pipeline.py'), Path('experiments/portfolio_chronos.py'),
             Path('experiments/chronos_experiment.py')]
    weights = {str(p.relative_to(args.model_path)): sha(p) for p in sorted(Path(args.model_path).glob('*.safetensors'))}
    if not weights:
        raise ValueError('Model weights missing')
    if weights != {'model.safetensors':WEIGHTS_SHA256}:
        raise ValueError('Weights differ from pinned HuggingFace blob')
    metadata=Path(args.model_path)/'.cache/huggingface/download/model.safetensors.metadata'
    if metadata.exists() and metadata.read_text().splitlines()[:2] != [REVISION,WEIGHTS_SHA256]:
        raise ValueError('Local HuggingFace revision/blob metadata mismatch')
    spec = dict(protocol='M20260927 v1', origins=ORIGINS, horizons=HORIZONS, trials=0,
        job_id=os.environ['SLURM_JOB_ID'], partition=os.environ['SLURM_JOB_PARTITION'],
        revision=REVISION, model_weights=weights, quantile=.5, seed=42, dtype='float32', context=512,
        batch_size=32, cross_learning=False, node=platform.node(), gpu=torch.cuda.get_device_name(),
        cpu_affinity=sorted(os.sched_getaffinity(0)), sha256={str(p): sha(p) for p in files},
        versions={p: importlib.metadata.version(p) for p in ['torch', 'chronos-forecasting', 'numpy', 'pandas']})
    write(root / 'run_started.json', spec)
    model = Chronos2Pipeline.from_pretrained(args.model_path, device_map='cuda', torch_dtype=torch.float32)
    model.model.eval()
    def predictor(tasks, horizon, cross):
        assert not cross
        with torch.inference_mode():
            quantiles, _ = model.predict_quantiles(tasks, prediction_length=horizon, quantile_levels=[.5],
                batch_size=32, context_length=512, cross_learning=False)
        return np.stack([q.cpu().numpy().reshape(horizon) for q in quantiles])
    for origin in ORIGINS:
        for mode in ['frozen', 'weekly']:
            for method in ['C0', 'C1']:
                parts = []
                for cutoff, end in segments(origin, mode):
                    folder = root / method / mode / origin / str(cutoff.date())
                    if time.monotonic() - started > 1600:
                        write(root / 'budget_stop.json', dict(seconds=time.monotonic()-started, next=str(folder)))
                        return
                    train = history.loc[history.date.le(cutoff)].copy(); t0 = time.monotonic()
                    try:
                        inference_end = cutoff + pd.Timedelta(days=61)
                        raw = forecast(history, method, cutoff, inference_end, predictor)
                        torch.cuda.synchronize()
                        meta = dict(method=method, mode=mode, origin=origin, cutoff=str(cutoff.date()), end=str(inference_end.date()),
                            emitted_end=str(end.date()), inference_horizon=61,
                            elapsed_seconds=time.monotonic()-t0, history_sha256=sha(args.history), code_sha256=sha(__file__),
                            revision=REVISION if method == 'C1' else None, model_weights=weights if method == 'C1' else {},
                            fact_availability='<=cutoff', job_id=os.environ['SLURM_JOB_ID'], seed=42)
                        p = checkpoint(train, raw, folder, meta)
                        p = p.loc[p.date.le(end)].copy()
                        p['prediction_cutoff'] = cutoff; p['actual_lead'] = (p.date-cutoff).dt.days
                        assert p.actual_lead.between(1, 7 if mode == 'weekly' else 61).all()
                        parts.append(p)
                    except (ValueError, RuntimeError) as error:
                        write(folder / 'error.json', dict(error=repr(error), cutoff=str(cutoff), end=str(end)))
                        print('ERROR', folder, repr(error), flush=True); break
                if len(parts) == len(list(segments(origin, mode))):
                    write(root / method / mode / origin / 'forecast.csv', pd.concat(parts).sort_values(KEYS))
                    print('completed', method, mode, origin, flush=True)
    write(root / 'completed.json', dict(job_id=os.environ['SLURM_JOB_ID'], seconds=time.monotonic()-started,
        process_cpu_seconds=time.process_time()-cpu_started, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def movement_matched(history, cutoff, end, available):
    train = history.loc[history.date.le(cutoff)].copy()
    mask = np.zeros(len(train), bool)
    for event in available:
        mask |= disrupted(train, event)
    normal = train.loc[~mask].copy()
    raw = ridge_forecast(normal, cutoff, end, PARAMS)
    daily = train.groupby(['route', 'date'], as_index=False).boardings.sum()
    ordinary = normal.groupby(['route', 'date'], as_index=False).boardings.sum().assign(weekday=lambda x:x.date.dt.dayofweek)
    profile = ordinary.groupby(['route', 'weekday']).boardings.median().clip(lower=1)
    for event in available:
        observed = daily.loc[disrupted(daily, event)].assign(weekday=lambda x:x.date.dt.dayofweek)
        observed = observed.merge(profile.rename('expected'), on=['route', 'weekday'], validate='many_to_one')
        observed['ratio'] = observed.boardings/observed.expected
        for route in [7, 50]:
            part = observed.loc[observed.route.eq(route)]
            factor = float(np.clip(part.ratio.median(), 0, 1)) if len(part) >= 3 else 1.
            if event == 'autumn' and route == 50: factor = 0.
            raw.loc[disrupted(raw, event) & raw.route.eq(route), 'prediction'] *= factor
    return raw


def run_cpu(args):
    if os.environ.get('SLURM_JOB_PARTITION') != 'ais-cpu' or len(os.sched_getaffinity(0)) > 2:
        raise RuntimeError('Expected bound ais-cpu allocation')
    root = Path(args.output); history = load_history(args.history); started=time.monotonic(); cpu=time.process_time()
    sources = [Path(args.notices_source)/'EVENTS.md', Path(args.notices_source)/'september.html',
               Path('artifacts/portfolio_20260926/ridge_probe/ridge/selection.json')]
    write(root/'run_started.json',dict(params=PARAMS,events=EVENTS,source_sha256={str(p):sha(p) for p in sources},
        job_id=os.environ['SLURM_JOB_ID'], selection='fixed historical P10 parameters; hindsight selected through2025-08-30',
        information='P11 original projected July window; restoration date unavailable until published', trials=0))
    for origin in ORIGINS:
        cutoff=pd.Timestamp(origin)-pd.Timedelta(days=1);end=cutoff+pd.Timedelta(days=61)
        for mode in ['no_notices','published_by_cutoff','all_hindsight']:
            available=[] if mode=='no_notices' else list(EVENTS) if mode=='all_hindsight' else [e for e,d in EVENTS.items() if pd.Timestamp(d)<=cutoff]
            raw=movement_matched(history,cutoff,end,available)
            checkpoint(history.loc[history.date.le(cutoff)],raw,root/mode/origin,
                dict(method='P10/P11',mode=mode,origin=origin,cutoff=str(cutoff.date()),end=str(end.date()),
                     available_events=available,parameter_selection_hindsight=True,history_sha256=sha(args.history),code_sha256=sha(__file__)))
    write(root/'completed.json',dict(seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,job_id=os.environ['SLURM_JOB_ID']))


def summarize(args):
    started=time.monotonic();cpu_started=time.process_time()
    history=load_history(args.history); root=Path(args.output); rows=[]; details=[]; missing=[]
    destination=Path(args.summary_output) if args.summary_output else root
    def unavailable(method,mode,origin,reason):
        missing.append(dict(method=method,mode=mode,origin=origin,reason=reason))
        for horizon in HORIZONS:
            for level in ['hourly','daily','route_calendar_month']:
                rows.append(dict(method=method,mode=mode,origin=origin,horizon=horizon,level=level,
                    status='NA',reason=reason,n=None,wape=None,wape_score=None,mae=None,absolute_error=None,
                    actual_total=None,predicted_total=None,bias=None,bias_ratio=None))
    for method in ['C0','C1']:
        for mode in ['frozen','weekly']:
            for origin in ORIGINS:
                path=root/'m1'/method/mode/origin/'forecast.csv'
                if not path.exists():
                    unavailable(method,mode,origin,'incomplete or failed computation');continue
                p=pd.read_csv(path,sep=';',parse_dates=['date'])
                r,d=evaluate(history,p,dict(method=method,mode=mode,origin=origin,information='cutoff inputs; modern pretrained for C1'))
                rows.extend(r);details.extend(d)
    sources={'031':'adaptive_shape/study/adaptive_shape/selected','029':'school_fraction/study/school_fraction/selected',
             '025':'bayes_shape_timesfm_blend/bayes_shape_timesfm_blend/selected'}
    for method,source in sources.items():
        for origin in ORIGINS:
            cutoff=str((pd.Timestamp(origin)-pd.Timedelta(days=1)).date())
            path=Path('artifacts/portfolio_20260926/continuation')/source/f'raw_{cutoff}.csv' if cutoff in ['2025-04-30','2025-06-30','2025-07-31','2025-08-31'] else Path('artifacts/selection_audit_20260927/new')/method/f'raw_{cutoff}.csv'
            if not path.exists():unavailable(method,'retrospective',origin,'raw unavailable');continue
            raw=pd.read_csv(path,sep=';',parse_dates=['date'],float_precision='round_trip')
            p=postprocess(raw);end=pd.Timestamp(cutoff)+pd.Timedelta(days=61)
            pd.testing.assert_frame_equal(p[KEYS],forecast_keys(cutoff,end))
            write(destination/'retrospective'/method/origin/'lineage.json',dict(source=str(path),sha256=sha(path),cutoff=cutoff,
                target_end=str(end.date()),information='retrospective external inputs; parameter selection hindsight',refitted=False))
            write(destination/'retrospective'/method/origin/'published.csv',p)
            r,d=evaluate(history,p,dict(method=method,mode='retrospective',origin=origin,information='retrospective+hindsight'))
            rows.extend(r);details.extend(d)
    for mode in ['no_notices','published_by_cutoff','all_hindsight']:
        for origin in ORIGINS:
            path=root/'m2'/mode/origin/'published.csv'
            if not path.exists():unavailable('P10/P11',mode,origin,'not completed');continue
            r,d=evaluate(history,pd.read_csv(path,sep=';',parse_dates=['date']),dict(method='P10/P11',mode=mode,origin=origin,information='fixed params selected later; coefficient inputs cutoff-safe'))
            rows.extend(r);details.extend(d)
    scores=pd.DataFrame(rows);write(destination/'metrics.csv',scores);write(destination/'breakdown.csv',pd.DataFrame(details));write(destination/'missing.csv',pd.DataFrame(missing))
    summary=[]
    for (method,mode,horizon,level),g in scores.groupby(['method','mode','horizon','level']):
        scopes=[('all_planned9',g,len(ORIGINS))]
        if mode=='retrospective' or method in ['C0','C1'] and mode=='frozen':
            shared_origins=[str((pd.Timestamp(c)+pd.Timedelta(days=1)).date()) for c in OLD]
            scopes.append(('shared_existing5',g.loc[g.origin.isin(shared_origins)],len(OLD)))
        for scope,part,expected in scopes:
            complete=len(part)==expected and part.absolute_error.notna().all()
            summary.append(dict(method=method,mode=mode,horizon=horizon,level=level,scope=scope,expected_origins=expected,
                available_origins=int(part.absolute_error.notna().sum()),complete=complete,
                mean_score=part.wape_score.mean() if complete else None,minimum_score=part.wape_score.min() if complete else None,
                maximum_score=part.wape_score.max() if complete else None,
                score_range=part.wape_score.max()-part.wape_score.min() if complete else None,
                pooled_wape=part.absolute_error.sum()/part.actual_total.sum() if complete and part.actual_total.sum() else None,
                pooled_score=max(0,1-part.absolute_error.sum()/part.actual_total.sum()) if complete and part.actual_total.sum() else None,
                pooled_note='repeated overlapping keys; not independent samples'))
    write(destination/'summary.csv',pd.DataFrame(summary))
    write(destination/'evaluator_run.json',dict(seconds=time.monotonic()-started,
        process_cpu_seconds=time.process_time()-cpu_started,code_sha256=sha(__file__),
        dataset_sha256=sha(args.history),metrics_rows=len(scores),breakdown_rows=len(details),
        numpy=np.__version__,pandas=pd.__version__,rounding='half-up once; aggregate final hours',
        overlapping_windows=True,independent_test=False))


def aggregate_export(args):
    history=load_history(args.history);root=Path(args.output);out=Path(args.summary_output)
    scores=pd.read_csv(out/'metrics.csv',sep=';');daily=[];monthly=[]
    specs=scores.loc[scores.absolute_error.notna(),['method','mode','origin']].drop_duplicates()
    for spec in specs.to_dict('records'):
        method,mode,origin=spec['method'],spec['mode'],spec['origin']
        path=(root/'m1'/method/mode/origin/'forecast.csv' if method in ['C0','C1'] else
              root/'m2'/mode/origin/'published.csv' if method=='P10/P11' else out/'retrospective'/method/origin/'published.csv')
        p=pd.read_csv(path,sep=';',parse_dates=['date'])
        joined=history[KEYS+['boardings']].merge(p[KEYS+['prediction']],on=KEYS,validate='one_to_one')
        assert len(joined)==14640
        for horizon in HORIZONS:
            group=joined.loc[joined.date.lt(pd.Timestamp(origin)+pd.Timedelta(days=horizon))].copy()
            group['month']=group.date.dt.strftime('%Y-%m')
            day=group.groupby(['route','date'],as_index=False)[['boardings','prediction']].sum()
            day=day.assign(**spec,horizon=horizon);daily.append(day)
            month=group.groupby(['route','month'],as_index=False)[['boardings','prediction']].sum()
            dates=group.groupby(['route','month']).date.nunique().rename('included_days').reset_index()
            month=month.merge(dates,on=['route','month'],validate='one_to_one')
            month['calendar_days']=pd.to_datetime(month.month+'-01').dt.days_in_month
            month['partial_month']=month.included_days.ne(month.calendar_days)
            monthly.append(month.assign(**spec,horizon=horizon))
    write(out/'daily_aggregates.csv',pd.concat(daily,ignore_index=True))
    write(out/'route_month_aggregates.csv',pd.concat(monthly,ignore_index=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['gpu','cpu','summarize','aggregates']);parser.add_argument('--history',default='artifacts/hourly_clean.csv')
    parser.add_argument('--output',required=True);parser.add_argument('--model-path')
    parser.add_argument('--notices-source',default='artifacts/portfolio_20260926/external')
    parser.add_argument('--summary-output')
    args=parser.parse_args();{'gpu':run_gpu,'cpu':run_cpu,'summarize':summarize,'aggregates':aggregate_export}[args.command](args)
