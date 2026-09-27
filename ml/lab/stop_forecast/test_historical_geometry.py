"""Small synthetic reader checks; no claim about Moscow stop accuracy."""
from copy import deepcopy
from historical_geometry import occurrences,PATTERNS

data={'osm3s':{'copyright':'test'},'elements':[]}
for rid,(_,origin,destination) in PATTERNS.items():
    data['elements'].append(dict(type='relation',id=rid,tags=dict(ref='12',route='tram',**{'from':origin,'to':destination}),
        members=[dict(type='node',ref=1,role='stop_entry_only'),dict(type='node',ref=1,role='stop_exit_only')]))
data['elements'].append(dict(type='node',id=1,lat=55.7,lon=37.7))
f=occurrences(data,'2025-09-15T00:00:00Z')
assert len(f)==4 and f.occurrence_id.is_unique and f.source_stop_id.nunique()==1
assert f.boarding_permitted_by_osm.sum()==2 and not f.actual_service_verified.any()
for kind in ['remark','missing_node','invalid_coordinate','metadata']:
    bad=deepcopy(data)
    if kind=='remark':bad['remark']='partial query'
    elif kind=='missing_node':bad['elements'].pop()
    elif kind=='invalid_coordinate':bad['elements'][-1]['lat']=float('nan')
    else:bad['elements'][-1]['uid']=123
    try:occurrences(bad,'2025-09-15T00:00:00Z')
    except ValueError:pass
    else:raise AssertionError(kind)
print('Repeated occurrences, entry/exit roles and invalid source rejection passed.')
