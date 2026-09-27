from copy import deepcopy
from prepare_live_pilot import positions

stop={'id':'00000000-0000-0000-0000-000000000001','num':1,'name':'A','lat':55.,'lon':37.}
data={'id':'00000000-0000-0000-0000-000000000002','number':'12','type':'tram','directions':[{'routePaths':[
    {'id':'00000000-0000-0000-0000-000000000003','stops':[stop,dict(stop,num=2)]},
    {'id':'00000000-0000-0000-0000-000000000004','stops':[dict(stop,name='B')]}]}]}
rows=positions(data)
assert len(rows)==3 and len({r['occurrence_id'] for r in rows})==3
assert len({r['source_direction_group'] for r in rows})==1 and len({r['source_pattern_id'] for r in rows})==2
assert all(r['observed_boardings'] is None for r in rows)
for kind in ['bus','sequence','coordinate','duplicate_pattern']:
    bad=deepcopy(data)
    if kind=='bus':bad['type']='bus'
    elif kind=='sequence':bad['directions'][0]['routePaths'][0]['stops'][1]['num']=1
    elif kind=='coordinate':bad['directions'][0]['routePaths'][0]['stops'][0]['lat']=float('nan')
    else:bad['directions'][0]['routePaths'][1]['id']=bad['directions'][0]['routePaths'][0]['id']
    try:positions(bad)
    except ValueError:pass
    else:raise AssertionError(kind)
print('Repeated stops, direction groups versus patterns, and source guards passed.')
