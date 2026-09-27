"""Fresh-process CPU inference, exact submission format and model integrity check."""
import argparse
import csv
import json
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time

import pandas as pd
from ml.lab.dense_student.model import load_bundle,predict,features
from ml.lab.cpu_student.model import read_history
from ml.lab.cpu_student.train import digest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--history',type=Path,required=True);a=p.parse_args()
    tick=time.perf_counter();model,meta=load_bundle(a.run/'bundle');history=read_history(a.history)
    loading=time.perf_counter()-tick;tick=time.perf_counter()
    pred,info=predict(model,meta,history,'2025-11-01');inference=time.perf_counter()-tick
    expected=pd.read_csv(a.run/'submission.csv',sep=';',parse_dates=['date'])
    pd.testing.assert_frame_equal(pred,expected)
    with (a.run/'submission.csv').open(newline='') as stream:
        reader=csv.DictReader(stream,delimiter=';');assert reader.fieldnames==['route','date','hour','prediction']
        rows=list(reader)
    assert len(rows)==14640 and all(r['prediction'].isdigit() for r in rows)
    assert len({(r['route'],r['date'],r['hour']) for r in rows})==14640
    assert all(r['prediction']=='0' for r in rows if r['route']=='5' or 1<=int(r['hour'])<=4)
    if meta.get('teacher_context_feature',False):
        for start,count in [('2025-08-28',5),('2025-08-31',6),('2025-09-01',6),('2025-11-01',8)]:
            _,x=features(history,start,meta)
            assert x.completed_teacher_windows.eq(count).all()
    with tempfile.TemporaryDirectory() as temporary:
        folder=Path(temporary)/'bundle';shutil.copytree(a.run/'bundle',folder)
        with (folder/meta['model_file']).open('ab') as stream:stream.write(b'corrupt')
        try:load_bundle(folder)
        except ValueError as e:assert 'checksum' in str(e).lower()
        else:raise AssertionError('Corrupt model accepted')
    modules=[n for n in ['torch','timesfm','tabpfn'] if n in sys.modules];assert not modules
    result=dict(load_model_history_seconds=loading,features_predict_seconds=inference,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
        model_bytes=(a.run/'bundle'/meta['model_file']).stat().st_size,exact_submission_reload=True,
        csv_format=True,checksum_corruption_rejected=True,gpu_modules_loaded=modules,
        submission_sha256=digest(a.run/'submission.csv'),forecast_info=info)
    (a.run/'portable_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
