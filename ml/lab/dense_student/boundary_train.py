"""Fixed-parameter diagnostic follow-up: dense teachers plus calibration-boundary origins."""
import argparse
import json
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

import pandas as pd
import numpy as np
from ml.lab.dense_student.train import LAB, context, examples, fit, save_bundle
from ml.lab.dense_student.model import load_bundle, predict
from ml.lab.cpu_student.model import read_history, grid, rounded
from ml.lab.cpu_student.train import digest, save_json, measure, SELECTED
from ml.lab.cpu_forecast.train import evaluated
from ml.runtime.constants import KEYS


def assemble(folder,boundaries):
    folder.mkdir(parents=True,exist_ok=False)
    teachers=folder/'teachers';teachers.mkdir()
    previous=LAB/'artifacts/dense_teacher_20260927_v2'
    for source in [previous,boundaries]:
        for path in (source/'teachers').glob('*'):
            shutil.copyfile(path,teachers/path.name)
        for path in source.glob('started_*.json'):
            shutil.copyfile(path,folder/path.name)
    imported={}
    for origin in ['2025-06-30','2025-07-31','2025-08-31','2025-10-31']:
        source=SELECTED/f'raw_{origin}.csv';dest=teachers/source.name
        shutil.copyfile(source,dest);imported[str(source)]=digest(source)
        save_json(dest.with_suffix('.json'),dict(origin=origin,end=str((pd.Timestamp(origin)+pd.Timedelta(days=61)).date()),
            history_max=origin,sha256=digest(dest),rows=14640,source_manifest='imported_monthly.json',
            history_input_physically_truncated=False,
            provenance='Existing frozen 030; history_max denotes fitting cutoff used by cutoff-guarded legacy functions, not original raw input extent'))
    save_json(folder/'imported_monthly.json',dict(source_sha256=imported,recipe='original frozen 030'))
    expected={str(d.date()) for d in pd.date_range('2025-04-09','2025-10-29',freq='7D')}|{
        '2025-05-31','2025-06-30','2025-07-31','2025-08-30','2025-08-31','2025-09-30','2025-10-31'}
    assert {p.stem[4:] for p in teachers.glob('*.csv')}==expected and len(expected)==37
    return teachers


def run(out,boundaries):
    begun=time.perf_counter();out.mkdir(parents=True,exist_ok=True)
    if (out/'bundle').exists():raise FileExistsError('Completed bundle exists')
    teacher_dir=assemble(out/'teacher_dataset',boundaries)
    code=Path(__file__).parent
    sources=[Path(__file__),code/'model.py',code/'train.py',out/'PROTOCOL.md',
        LAB/'artifacts/dense_student_20260927_v2/selection.json',LAB/'artifacts/hourly_clean.csv',
        *sorted(teacher_dir.glob('*'))]
    hashes={str(p):digest(p) for p in sources}
    selected=json.loads((LAB/'artifacts/dense_student_20260927_v2/selection.json').read_text())['config']
    assert selected==dict(kind='catboost',alpha=.5)
    common=dict(context(),teacher_context_feature=True,include_exact_origin=False)
    save_json(out/'protocol.json',dict(config=selected,sources_sha256=hashes,context=common,
        diagnostic_test_already_viewed=True,final_exact_november_teacher_in_fit=True,budget_seconds=600))
    history=read_history(LAB/'artifacts/hourly_clean.csv')
    actual,soft,meta,audit_test=examples(history,'2025-08-31',common,teacher_dir)
    assert soft.origin.max()==pd.Timestamp('2025-08-30') and actual.target_date.max()<=pd.Timestamp('2025-08-31')
    model,info=fit(actual,soft,meta,selected);meta=save_bundle(out/'evaluation',model,info)
    pred,_=predict(model,meta,history,'2025-09-01');paired,metric=evaluated(history,pred)
    pred.to_csv(out/'test_september_october.csv',sep=';',index=False,date_format='%Y-%m-%d')
    rows=[dict(model='boundary_student',**metric)];print('DIAGNOSTIC TEST',metric,flush=True)
    old=pd.read_csv(LAB/'artifacts/dense_student_20260927_v2/test_metrics.csv',sep=';')
    pd.concat([pd.DataFrame(rows),old],ignore_index=True).to_csv(out/'test_metrics.csv',sep=';',index=False)
    details=[]
    paired['week']=(paired.date-pd.Timestamp('2025-09-01')).dt.days//7+1
    for dimension in ['route','hour','week']:
        for value,group in paired.groupby(dimension):details.append(dict(dimension=dimension,value=int(value),**measure(group.boardings,group.prediction)))
    pd.DataFrame(details).to_csv(out/'breakdown.csv',sep=';',index=False)
    teacher=pd.read_csv(SELECTED/'raw_2025-08-31.csv',sep=';',parse_dates=['date']).sort_values(KEYS).reset_index(drop=True)
    teacher=rounded(teacher[KEYS],teacher.prediction)
    fidelity=[dict(start='2025-09-01',role='heldout_exact_origin_diagnostic',**measure(teacher.prediction,pred.prediction))]
    common['include_exact_origin']=True
    actual,soft,meta,audit_final=examples(history,'2025-10-31',common,teacher_dir)
    assert soft.origin.max()==pd.Timestamp('2025-10-31') and actual.target_date.max()==pd.Timestamp('2025-10-31')
    tick=time.perf_counter();model,info=fit(actual,soft,meta,selected);fit_seconds=time.perf_counter()-tick
    meta=save_bundle(out/'bundle',model,info)
    loaded,meta=load_bundle(out/'bundle')
    tick=time.perf_counter();submission,info=predict(loaded,meta,history,'2025-11-01');inference=time.perf_counter()-tick
    direct,_=predict(model,meta,history,'2025-11-01');pd.testing.assert_frame_equal(submission,direct)
    pd.testing.assert_frame_equal(submission[KEYS],grid('2025-11-01'))
    submission.to_csv(out/'submission.csv',sep=';',index=False,date_format='%Y-%m-%d')
    for start,path,role in [('2025-11-01',LAB.parent/'bundles/030/forecast.csv','training_fidelity_exact_teacher_in_fit'),
                           ('2025-08-15',LAB/'artifacts/selection_audit_20260927/new/030/raw_2025-08-14.csv','retrospective_year_interpolation_heldout_origin')]:
        teacher=pd.read_csv(path,sep=';',parse_dates=['date']).sort_values(KEYS).reset_index(drop=True)
        teacher=rounded(teacher[KEYS],teacher.prediction)
        prediction,_=predict(loaded,meta,history,start)
        pd.testing.assert_frame_equal(teacher[KEYS],prediction[KEYS])
        fidelity.append(dict(start=start,role=role,**measure(teacher.prediction,prediction.prediction)))
    pd.DataFrame(fidelity).to_csv(out/'teacher_fidelity.csv',sep=';',index=False)
    for start in ['2025-01-01','2025-09-15','2025-12-31']:
        prediction,_=predict(loaded,meta,history,start)
        pd.testing.assert_frame_equal(prediction[KEYS],grid(start))
        assert prediction.prediction.dtype==np.dtype('int64') and prediction.prediction.ge(0).all()
        assert prediction.loc[prediction.route.eq(5)|prediction.hour.between(1,4),'prediction'].eq(0).all()
    before,_=predict(loaded,meta,history,'2025-09-15')
    poisoned=history.copy();future=poisoned.date.ge('2025-09-15')&poisoned.route.ne(5)&~poisoned.hour.between(1,4)
    poisoned.loc[future,'boardings']=123456789;poisoned.loc[future,'working_events_observed']=False
    after,_=predict(loaded,meta,poisoned,'2025-09-15');pd.testing.assert_frame_equal(before,after)
    if any(digest(Path(p))!=h for p,h in hashes.items()):raise ValueError('Inputs changed during run')
    save_json(out/'training_audit.json',dict(test=audit_test,final=audit_final))
    save_json(out/'verification.json',dict(model_reload_exact=True,full_grid=True,nonnegative_int64=True,
        future_target_mask_poison_unchanged=True,structural_zeros=True,year_boundary_requests=True,
        sources_unchanged=True,gpu_modules_loaded=[n for n in ['torch','tabpfn','timesfm'] if n in sys.modules],
        submission_sha256=digest(out/'submission.csv'),rows=len(submission)))
    save_json(out/'completed.json',dict(seconds=time.perf_counter()-begun,final_fit_seconds=fit_seconds,
        inference_seconds=inference,model_bytes=(out/'bundle'/meta['model_file']).stat().st_size,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)))
    print('COMPLETE',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--boundaries',type=Path,required=True);a=p.parse_args();signal.alarm(600)
    run(a.output.resolve(),a.boundaries.resolve())
