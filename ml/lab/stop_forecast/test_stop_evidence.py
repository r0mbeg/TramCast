"""Prevent telemetry predictions, missing values and zero ETAs from becoming labels."""
from copy import deepcopy
from collect_stop_evidence import normalize

stop='99a2734d-84ed-4361-8cc2-5cb3e4e778c5'
data=dict(id=stop,lat=55.7,lon=37.7,routePath=[dict(id='route-x',number='12',type='tram',
    externalForecast=[dict(time=0,byTelemetry=1,tmId=7),dict(time=90,byTelemetry=0)])])
rows=normalize(data,stop,'2026-09-27T12:00:00+00:00')
assert len(rows)==2 and [r['evidence_kind'] for r in rows]==['telemetry_eta','schedule_eta']
assert all(r['observed_boardings'] is None and r['actual_arrival_time'] is None and not r['usable_as_stop_truth'] for r in rows)
empty=deepcopy(data);empty['routePath'][0]['externalForecast']=[]
assert normalize(empty,stop,'2026-09-27T12:00:00+00:00')==[]
for kind in ['missing_paths','missing_eta','negative_eta','unknown_flag','wrong_stop','bad_coords']:
    bad=deepcopy(data)
    if kind=='missing_paths':del bad['routePath']
    elif kind=='missing_eta':del bad['routePath'][0]['externalForecast'][0]['time']
    elif kind=='negative_eta':bad['routePath'][0]['externalForecast'][0]['time']=-1
    elif kind=='unknown_flag':bad['routePath'][0]['externalForecast'][0]['byTelemetry']=2
    elif kind=='wrong_stop':bad['id']='other'
    else:bad['lat']=float('nan')
    try:normalize(bad,stop,'2026-09-27T12:00:00+00:00')
    except ValueError:pass
    else:raise AssertionError(kind)
print('ETA semantics, zero/missing distinction and invalid-source checks passed.')
