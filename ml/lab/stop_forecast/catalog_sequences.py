"""Read application geography without treating it as historical boarding truth."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from audit import ROOT,prior
from build_metro_access import distances


def catalog():
    path=ROOT/'data/catalog/catalog.xlsx'
    tables={name:rows for name,_,_,rows in prior.sheets(path)}
    originals=[]
    for original in sorted((ROOT/'dataset/spravochniki').glob('*.xlsx')):
        old={name:rows for name,_,_,rows in prior.sheets(original) if name in tables}
        if len(old)==3:
            originals.append((original,{name:rows==old[name] for name,rows in tables.items()}))
    if len(originals)!=1 or not all(originals[0][1].values()):
        raise ValueError('Application catalog differs; inspect before use')
    routes=pd.DataFrame(tables['Маршруты GTFS_ROUTES'])
    stops=pd.DataFrame(tables['Остановки GTFS_STOPS'])
    positions=pd.DataFrame(tables['Порядок_остановок GTFS_TRIPS_ST'])
    if stops.stop_id.duplicated().any() or routes.route_id.duplicated().any():
        raise ValueError('Duplicate catalog identifiers')
    f=positions.merge(stops[['stop_id','stop_name','stop_lat','stop_lon']],on='stop_id',how='left',validate='many_to_one')
    for c in ['route_short_name','stop_sequence','direction_id','stop_lat','stop_lon']:
        f[c]=pd.to_numeric(f[c],errors='raise')
    if f[['stop_lat','stop_lon']].isna().any().any() or f.duplicated(['route_id','trip_id','stop_sequence']).any():
        raise ValueError('Invalid occurrence support')
    f=f.rename(columns={'route_short_name':'route'}).sort_values(['route','direction_id','trip_id','stop_sequence']).reset_index(drop=True)
    f['source_stop_id']=f.stop_id
    f['source_pattern_id']=f.trip_id
    f['occurrence_id']='catalog:'+f.route.astype(str)+':'+f.trip_id+':'+f.stop_sequence.astype(str)
    f['historical_validity_verified']=False
    osm_path=ROOT/'data/osm/tram_routes.json'
    osm=json.loads(osm_path.read_text());elements=osm['elements']
    nodes={e['id']:e for e in elements if e['type']=='node'}
    relations=[e for e in elements if e['type']=='relation' and e.get('tags',{}).get('type')=='route']
    roles={'stop','stop_entry_only','stop_exit_only'}
    refs=[m['ref'] for e in relations for m in e['members'] if m['type']=='node' and m['role'] in roles]
    if not set(refs)<=nodes.keys():
        raise ValueError('OSM referenced stop missing')
    summary=dict(catalog_routes=len(routes),catalog_stops=len(stops),catalog_occurrences=len(f),
        catalog_patterns=int(f.trip_id.nunique()),sheets_equal_to_original=originals[0][1],
        repeated_occurrences_within_pattern=int(f.duplicated(['trip_id','stop_id']).sum()),
        osm_routes=sorted({e['tags']['ref'] for e in relations},key=int),osm_patterns=len(relations),
        osm_stop_occurrences=len(refs),osm_referenced_stop_nodes=len(set(refs)),
        osm_all_nodes=len(nodes),osm_ways=sum(e['type']=='way' for e in elements),
        osm_snapshot= osm['osm3s']['timestamp_osm_base'],
        stop_event_observations=0,scheduled_or_actual_stop_times=0)
    return f,summary,[path,originals[0][0],osm_path]


def pilot_cycle(f):
    p=f.loc[f.route.eq(12)].copy().reset_index(drop=True)
    if p.groupby('direction_id').trip_id.nunique().to_dict()!={0:1,1:1}:
        raise ValueError('Expected two route12 catalog patterns')
    if p.end_date.fillna('').ne('').any() or p.is_addpoint.ne('0').any():
        raise ValueError('Review inactive/additional points')
    nxt=np.roll(np.arange(len(p)),-1)
    length=np.diag(distances(p.stop_lat,p.stop_lon,p.stop_lat.iloc[nxt],p.stop_lon.iloc[nxt]))
    turn=p.direction_id.to_numpy()!=p.direction_id.to_numpy()[nxt]
    p['edge_to_occurrence_id']=p.occurrence_id.to_numpy()[nxt]
    p['edge_straight_distance_m']=length
    p['edge_is_assumed_turnaround']=turn
    return p,length,turn
