"""One bounded capture of public ETA responses; never converts ETA to actual visits."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
from urllib.error import HTTPError,URLError
from urllib.request import Request,urlopen
from uuid import UUID

BASE='https://moscowtransport.app/api/stop_v2/'
HERE=Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def normalize(data,stop_id,captured_at):
    if str(UUID(stop_id))!=stop_id or data.get('id')!=stop_id:
        raise ValueError('Stop identifier mismatch')
    when=datetime.fromisoformat(captured_at)
    if when.tzinfo is None:
        raise ValueError('Capture timestamp must have timezone')
    for key,limit in [('lat',90),('lon',180)]:
        value=data.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or abs(value)>limit:
            raise ValueError('Invalid stop coordinates')
    paths=data.get('routePath')
    if not isinstance(paths,list):
        raise ValueError('Missing routePath; do not interpret as no service')
    rows=[]
    for route in paths:
        if not all(route.get(k) for k in ['id','number','type']):
            raise ValueError('Incomplete route identity')
        forecasts=route.get('externalForecast')
        if not isinstance(forecasts,list):
            raise ValueError('Missing forecast array; do not turn into zero')
        for item in forecasts:
            seconds=item.get('time');flag=item.get('byTelemetry')
            if type(seconds) is not int or seconds<0 or type(flag) is not int or flag not in [0,1]:
                raise ValueError('Unknown ETA units/value or telemetry flag')
            rows.append(dict(captured_at_utc=captured_at,stop_uuid=stop_id,
                source_internal_stop_id=str(data.get('internal_id','')),stop_name=data.get('name'),
                stop_lat=data['lat'],stop_lon=data['lon'],source_route_id=route['id'],
                route=str(route['number']),transport_type=route['type'],destination=route.get('lastStopName'),
                eta_seconds=seconds,evidence_kind='telemetry_eta' if flag else 'schedule_eta',
                source_tm_id=item.get('tmId'),source_path_id=item.get('routePathId'),
                tm_id_vehicle_identity_verified=False,actual_arrival_time=None,
                observed_boardings=None,usable_as_stop_truth=False))
    return rows


def capture(stop_ids,out):
    if not stop_ids or len(stop_ids)!=len(set(stop_ids)) or len(stop_ids)>20:
        raise ValueError('Provide 1–20 distinct stops per bounded capture')
    for stop in stop_ids:
        if str(UUID(stop))!=stop:
            raise ValueError('Expected canonical stop UUID')
    out.mkdir(parents=True,exist_ok=False)
    (out/'raw').mkdir()
    requests=[];rows=[]
    for stop in stop_ids:
        started=datetime.now(timezone.utc).isoformat();record=dict(stop_uuid=stop,url=BASE+stop,started_at_utc=started)
        try:
            req=Request(BASE+stop,headers={'User-Agent':'TramCast-research/0.1','Accept':'application/json'})
            # ponytail: one sequential request per stop, no daemon/retries; schedule only after source/coverage validation.
            with urlopen(req,timeout=25) as response:
                body=response.read(2_000_001)
                record.update(http_status=response.status,server_date=response.headers.get('Date'),
                              content_type=response.headers.get('Content-Type'),final_url=response.url)
            if len(body)>2_000_000:
                raise ValueError('Response too large')
            ended=datetime.now(timezone.utc).isoformat()
            raw=out/'raw'/f'{stop}.json';raw.write_bytes(body)
            record.update(finished_at_utc=ended,raw_file=str(raw.relative_to(out)),sha256=digest(body),bytes=len(body))
            parsed=normalize(json.loads(body),stop,ended)
            rows.extend(parsed);record.update(status='captured_eta_only',forecast_rows=len(parsed))
        except (HTTPError,URLError,TimeoutError,OSError,ValueError,KeyError,TypeError) as error:
            record.update(status='failed',error_type=type(error).__name__,
                          http_status=getattr(error,'code',record.get('http_status')),
                          finished_at_utc=datetime.now(timezone.utc).isoformat())
        requests.append(record)
    write(out/'requests.json',requests)
    write(out/'eta_observations.json',rows)
    summary=dict(version='L20260927-v1',kind='prospective_eta_capture_not_boarding_dataset',
        requested_stops=len(stop_ids),successful_responses=sum(r['status']=='captured_eta_only' for r in requests),
        forecast_rows=len(rows),telemetry_eta_rows=sum(r['evidence_kind']=='telemetry_eta' for r in rows),
        schedule_eta_rows=sum(r['evidence_kind']=='schedule_eta' for r in rows),
        route12_rows=sum(r['route']=='12' and r['transport_type']=='tram' for r in rows),
        actual_stop_visits=0,successful_validation_labels=0,source_freshness_verified=False,
        dates=sorted({r['captured_at_utc'][:10] for r in rows}),historical_2025_coverage=False)
    write(out/'summary.json',summary)
    write(out/'manifest.json',dict(version=summary['version'],code_sha256=digest(Path(__file__).read_bytes()),
        outputs={str(p.relative_to(out)):digest(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file()},
        source=BASE,source_license='not established; local research only',stop_truth_available=False))
    print(json.dumps(summary,ensure_ascii=False))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stop-id',action='append',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if capture(args.stop_id,args.out)['successful_responses']!=len(args.stop_id):
        raise SystemExit('Incomplete capture; inspect requests.json. Missing responses are not zero observations.')
