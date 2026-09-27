"""Fixed fresh Sep15–21 device holdout using the frozen V selection rules."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from audit import ROOT, sha, valid_key, write_json
from device_signal import score

HERE=Path(__file__).resolve().parent


def run(out):
    raw=ROOT/'dataset/test.csv'
    before=raw.stat()
    cols=['tran_date_time','device_no','garage_number','ngpt_route','validation_result']
    parts=[]
    for f in pd.read_csv(raw,sep=';',dtype=str,usecols=cols,keep_default_na=False,chunksize=250000):
        mask=f.tran_date_time.ge('2025-09-15') & f.tran_date_time.lt('2025-09-22')
        if mask.any():
            parts.append(f.loc[mask].copy())
    f=pd.concat(parts,ignore_index=True)
    f['time']=pd.to_datetime(f.tran_date_time,format='%Y-%m-%d %H:%M:%S',errors='raise')
    f['date']=f.time.dt.strftime('%Y-%m-%d')
    valid=valid_key(f.device_no)&valid_key(f.garage_number)
    dv=f.loc[valid].groupby(['date','device_no']).garage_number.nunique()
    vr=f.loc[valid].groupby(['date','garage_number']).ngpt_route.nunique()
    dc=pd.MultiIndex.from_frame(f[['date','device_no']]).isin(dv.loc[dv.gt(1)].index)
    rc=pd.MultiIndex.from_frame(f[['date','garage_number']]).isin(vr.loc[vr.gt(1)].index)
    minutes=f.time.dt.hour*60+f.time.dt.minute
    keep=f.ngpt_route.eq('12 трамвай')&f.validation_result.eq('1')&(minutes.lt(60)|minutes.ge(330))
    total=int(keep.sum())
    ledger_path=HERE/'runs/20260927-v2/route_hour_balance.csv'
    ledger=pd.read_csv(ledger_path,sep=';')
    expected=ledger.loc[ledger.route.eq(12)&ledger.date.ge('2025-09-15')&ledger.date.lt('2025-09-22'),'boardings'].sum()
    if total!=expected:
        raise ValueError('Holdout ledger mismatch')
    exclusions={}
    for name,reject in [('invalid_keys',~valid),('conflicting_device_day',dc),
                        ('multiple_route_vehicle_day',rc),('partial_hour5',f.time.dt.hour.eq(5))]:
        exclusions[name]=int((keep&reject).sum());keep &= ~reject
    f=f.loc[keep].copy()
    f['hour']=f.time.dt.hour;f['bin']=(f.time.dt.minute*60+f.time.dt.second)//10
    rows,held,reference=[],[],[]
    for group_id,((day,_),g) in enumerate(f.groupby(['date','garage_number'],sort=True)):
        device=sorted(g.device_no.unique())[0]
        for hour,h in g.groupby('hour'):
            left=h.loc[h.device_no.eq(device),'bin'].to_numpy()
            right=h.loc[h.device_no.ne(device),'bin'].to_numpy()
            if len(left)<10 or len(right)<20:
                continue
            hb,rb=np.bincount(left,minlength=360),np.bincount(right,minlength=360)
            rows.append(dict(vehicle_day=group_id,date=day,hour=int(hour),**score(hb,rb)))
            held.append(hb);reference.append(rb)
    frame=pd.DataFrame(rows)
    if frame.empty:
        raise ValueError('No eligible holdout hours')
    used=int(frame.held_events.sum()+frame.reference_events.sum())
    exclusions['insufficient_device_hour_events']=len(f)-used
    assert used+sum(exclusions.values())==total
    old_manifest=HERE/'runs/device-signal-20260927-v1/manifest.json'
    raw_sha=sha(raw)
    if raw_sha!=json.loads(old_manifest.read_text())['inputs'][str(raw.relative_to(ROOT))]:
        raise ValueError('Raw source changed')
    after=raw.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('Raw changed during scan')
    out.mkdir(parents=True,exist_ok=False)
    frame.to_csv(out/'device_hour_scores.csv',sep=';',index=False)
    np.savez_compressed(out/'binned_counts.npz',held=np.asarray(held),reference=np.asarray(reference))
    write_json(out/'summary.json',dict(version='VH20260927-v1',period=['2025-09-15','2025-09-21'],
        pilot_clean_successes=total,used_events=used,held_events=int(frame.held_events.sum()),
        hours=len(frame),vehicle_days=int(frame.vehicle_day.nunique()),exclusions=exclusions,stop_labels=0))
    inputs=[Path(__file__),HERE/'device_signal.py',HERE/'audit.py',HERE/'ACTIVITY_PROTOCOL.md',ledger_path,old_manifest]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs};hashes[str(raw.relative_to(ROOT))]=raw_sha
    write_json(out/'manifest.json',dict(version='VH20260927-v1',inputs=hashes,
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print('Prepared fresh device holdout:',len(frame),'hours;',used,'events',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
