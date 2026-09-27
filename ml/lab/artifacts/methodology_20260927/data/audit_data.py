"""M3 metadata/aggregate audit; run from repository root, stdlib only.

Reads one raw header per file, never scans raw events. Writes this directory only.
"""
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import time
import xml.etree.ElementTree as ET
from zipfile import ZipFile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
TARGETS = (1, 5, 7, 11, 12, 17, 25, 26, 28, 50)
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
HISTORY_SHA = "7031c686c149fcb552711df49d82bab23540d8c80ffa6f140c6d0444138384ae"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(name, rows):
    with (OUT/name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def sheets(path):
    """Read actual XLSX XML values, preserving original headers/IDs as text."""
    with ZipFile(path) as z:
        strings = ["".join(x.itertext()) for x in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("s:si", NS)] if "xl/sharedStrings.xml" in z.namelist() else []
        links = {r.attrib["Id"]: r.attrib["Target"] for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
        book = ET.fromstring(z.read("xl/workbook.xml"))
        for sheet in book.find("s:sheets", NS):
            target = links[sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
            target = target.lstrip("/") if target.startswith("/") else "xl/"+target
            rows = []
            for row in ET.fromstring(z.read(target)).findall("s:sheetData/s:row", NS):
                values = {}
                for cell in row:
                    value = cell.find("s:v", NS)
                    value = "" if value is None else value.text
                    if cell.attrib.get("t") == "s":
                        value = strings[int(value)]
                    elif cell.attrib.get("t") == "inlineStr":
                        value = "".join(cell.find("s:is", NS).itertext())
                    values[re.sub(r"\d", "", cell.attrib["r"])] = value
                if values:
                    rows.append(values)
            header_index = 0 if sheet.attrib["name"] in ("Порядок_с_координатами", "Лист1") else 1
            header = rows[header_index]
            records = [{field: row.get(col, "") for col, field in header.items()} for row in rows[header_index+1:]]
            yield sheet.attrib["name"], header_index+1, list(header.values()), records


def coverage(rows):
    active = [r for r in rows if r["route"] != 5 and r["hour"] not in range(1, 5)]
    return dict(cells=len(rows), structural_route5=sum(r["route"] == 5 for r in rows),
                structural_night_non5=sum(r["route"] != 5 and r["hour"] in range(1, 5) for r in rows),
                active_cells=len(active), active_event_present=sum(r["working_events_observed"] for r in active),
                active_success_present=sum(r["successful_working_observed"] for r in active),
                active_no_events=sum(not r["working_events_observed"] for r in active),
                active_no_success=sum(not r["successful_working_observed"] for r in active),
                active_unsuccessful_only=sum(r["working_events_observed"] and not r["successful_working_observed"] for r in active),
                supplied_label_present=sum(r["supplied_label_present"] for r in rows),
                raw_total=sum(r["raw_boardings"] for r in rows), clean_total=sum(r["boardings"] for r in rows),
                working_events=sum(r["working_events"] for r in rows),
                complete_counter_cells=None)


def run():
    started = time.monotonic()
    cpu_before = time.process_time()
    inventory = []
    prior_audit = json.loads((ROOT/"ml/artifacts/preparation_audit.json").read_text())
    for name in ("train.csv", "test.csv"):
        path = ROOT/"dataset"/name
        with path.open(encoding="utf-8-sig", newline="") as f:
            headers = next(csv.reader(f, delimiter=";"))
        stat = path.stat()
        assert stat.st_size == prior_audit[name]["bytes"] and stat.st_mtime_ns == prior_audit[name]["mtime_ns"]
        inventory.append(dict(path=str(path.relative_to(ROOT)), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                              headers=headers, read_scope="first CSV header only", sha256=None,
                              stat_matches_prior_audit=True,
                              canonical_header_sha256=hashlib.sha256(json.dumps(headers).encode()).hexdigest(),
                              spatial_fields_absent=[x for x in ("stop_id", "direction_id", "trip_id", "latitude", "longitude", "alighting") if x not in headers]))
    history_path = ROOT/"ml/artifacts/hourly_clean.csv"
    assert sha(history_path) == HISTORY_SHA
    rows = read_csv(history_path)
    for r in rows:
        for c in ("route", "hour", "raw_boardings", "boardings", "working_events", "removed_boardings"):
            r[c] = int(r[c])
        for c in ("supplied_label_present", "successful_working_observed", "working_events_observed"):
            assert r[c] in ("True", "False")
            r[c] = r[c] == "True"
        assert r["boardings"] >= 0
        assert r["successful_working_observed"] == (r["boardings"] > 0)
        assert r["working_events_observed"] == (r["working_events"] > 0)
        assert r["raw_boardings"] - r["boardings"] == r["removed_boardings"]
    expected = {(route, (date(2025, 1, 1)+timedelta(days=d)).isoformat(), hour) for route in TARGETS for d in range(304) for hour in range(24)}
    truth = {(r["route"], r["date"], r["hour"]): r for r in rows}
    assert len(rows) == len(truth) == 72960 and set(truth) == expected
    labels = read_csv(ROOT/"dataset/labels/labels_day_train.csv")+read_csv(ROOT/"dataset/labels/labels_day_test.csv")
    label_keys = {(int(r["route"]), r["date"], int(r["hour"])): int(r["boardings"]) for r in labels}
    assert len(label_keys) == len(labels) == 57551
    assert all(r["raw_boardings"] == label_keys.get(k, 0) and r["supplied_label_present"] == (k in label_keys) for k, r in truth.items())
    summary = coverage(rows)
    assert summary["clean_total"] == 59545140 and summary["active_no_events"] == 730 and summary["active_no_success"] == 732
    summary.update(clean_sha256=HISTORY_SHA, labels_absent_cells=len(rows)-len(labels),
                   sample_start="2025-01-01", sample_end="2025-10-31", completeness="unknown even when events present")
    write_json("coverage_summary.json", summary)
    for name, columns in (("coverage_by_route.csv", ("route",)), ("coverage_by_route_hour.csv", ("route", "hour")), ("coverage_by_route_date.csv", ("route", "date"))):
        groups = defaultdict(list)
        for r in rows:
            groups[tuple(r[c] for c in columns)].append(r)
        write_csv(name, [dict(zip(columns, key), **coverage(group)) for key, group in sorted(groups.items())])
    missing = [r for r in rows if r["route"] != 5 and r["hour"] not in range(1, 5) and not r["successful_working_observed"]]
    write_csv("active_no_success_cells.csv", missing)
    missing50 = [r for r in rows if r["route"] == 50 and r["date"] == "2025-09-21"]
    assert len(missing50) == 24 and all(r["working_events"] == r["boardings"] == r["raw_boardings"] == 0 and not r["supplied_label_present"] for r in missing50)
    write_csv("route50_20250921.csv", missing50)
    geo = []
    route_records, stop_records, order_records = [], [], []
    for path in sorted((ROOT/"dataset/spravochniki").glob("*.xlsx")):
        for name, header_row, headers, records in sheets(path):
            temporal = {c: dict(Counter(r[c] for r in records)) for c in ("actual_date", "route_date_start", "start_date", "date") if c in headers}
            routes = sorted({int(r["route_short_name"]) for r in records if r.get("route_short_name", "").isdigit()})
            info = dict(path=str(path.relative_to(ROOT)), file_sha256=sha(path), sheet=name, header_row=header_row,
                        data_rows=len(records), headers=headers, routes=routes, target_overlap=sorted(set(routes)&set(TARGETS)), temporal_values=temporal)
            if name == "Маршруты GTFS_ROUTES": route_records = records
            if name == "Остановки GTFS_STOPS": stop_records = records
            if name == "Порядок_остановок GTFS_TRIPS_ST": order_records = records
            if "trip_id" in headers:
                info["unique_trips"] = len({r["trip_id"] for r in records})
                info["directions"] = sorted({r["direction_id"] for r in records})
            if "stop_lat" in headers:
                info["missing_coordinates"] = sum(not r["stop_lat"] or not r["stop_lon"] for r in records)
            if name == "Лист1":
                vals = [datetime(1899,12,30)+timedelta(days=float(r["Дата и время транзакции"])) for r in records]
                info["event_date_min"] = min(vals).isoformat()
                info["event_date_max"] = max(vals).isoformat()
            geo.append(info)
    route_numbers = {int(r["route_short_name"]) for r in route_records}
    stop_ids = {r["stop_id"] for r in stop_records}
    assert len(route_records) == 10 and len(stop_records) == len(stop_ids) == 489 and len(order_records) == 622
    assert all(r["stop_id"] in stop_ids for r in order_records)
    write_json("geography_inventory.json", dict(sheets=geo, target_routes=list(TARGETS),
               geography_routes=sorted(route_numbers), geography_target_overlap=sorted(route_numbers&set(TARGETS)),
               target_without_geography=sorted(set(TARGETS)-route_numbers),
               order_stop_references_valid=True, ordered_stop_count=622, unique_trip_templates=len({r["trip_id"] for r in order_records})))
    label_files = [ROOT/"dataset/labels"/f"labels_day_{split}.csv" for split in ("train", "test")]
    write_json("data_inventory.json", dict(raw_files=inventory, aggregate=dict(path=str(history_path.relative_to(ROOT)), sha256=HISTORY_SHA, rows=len(rows)),
               labels=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p), rows=len(read_csv(p))) for p in label_files],
               preparation_audit=dict(path="ml/artifacts/preparation_audit.json", sha256=sha(ROOT/"ml/artifacts/preparation_audit.json"), value=prior_audit),
               target="realized successful validations after clock-window cleaning", supply_and_observation_completeness="unknown"))
    paths = {"C0": "ml/artifacts/portfolio_20260926/cpu/mean_all", "C1": "ml/artifacts/portfolio_20260926/gpu/daily", "031": "ml/artifacts/portfolio_20260926/continuation/adaptive_shape/study/adaptive_shape/selected"}
    sensitivity, lineage = [], []
    for cutoff in ("2025-04-30", "2025-06-30", "2025-07-31", "2025-08-31"):
        end = (date.fromisoformat(cutoff)+timedelta(days=61)).isoformat()
        keys = {k for k in expected if cutoff < k[1] <= end}
        assert len(keys) == 14640
        for method, prefix in paths.items():
            path = ROOT/prefix/f"raw_{cutoff}.csv"
            predictions = read_csv(path)
            pred_keys = {(int(r["route"]), r["date"], int(r["hour"])) for r in predictions}
            assert pred_keys == keys and len(predictions) == len(pred_keys)
            values = {}
            for r in predictions:
                k = (int(r["route"]), r["date"], int(r["hour"]))
                value = float(r["prediction"])
                assert math.isfinite(value) and value >= 0
                values[k] = 0 if k[0] == 5 or k[2] in range(1,5) else math.floor(value+0.5)
            lineage.append(dict(method=method, cutoff=cutoff, path=str(path.relative_to(ROOT)), sha256=sha(path), rows=len(predictions), postprocess="pipeline.postprocess-equivalent half-up, structural zeros"))
            for mask in ("full_absent_as_zero", "legacy_observed_including_structural", "working_any_event_present", "working_success_present"):
                active = lambda k: k[0] != 5 and k[2] not in range(1,5)
                keep = [k for k in sorted(keys) if mask == "full_absent_as_zero" or
                        (mask == "legacy_observed_including_structural" and (not active(k) or truth[k]["working_events_observed"])) or
                        (mask == "working_any_event_present" and active(k) and truth[k]["working_events_observed"]) or
                        (mask == "working_success_present" and active(k) and truth[k]["successful_working_observed"])]
                actual = sum(truth[k]["boardings"] for k in keep)
                forecast = sum(values[k] for k in keep)
                error = sum(abs(truth[k]["boardings"]-values[k]) for k in keep)
                sensitivity.append(dict(method=method, cutoff=cutoff, end=end, mask=mask, rows=len(keep), full_grid_rows=len(keys),
                    event_present_working_rows=sum(active(k) and truth[k]["working_events_observed"] for k in keys),
                    working_rows=sum(active(k) for k in keys), actual_total=actual, predicted_total=forecast, absolute_error=error,
                    MAE=error/len(keep) if keep else None, WAPE=error/actual if actual else None,
                    score=max(0,1-error/actual) if actual else None, signed_bias=forecast-actual))
    write_csv("sensitivity_historical.csv", sensitivity)
    write_json("forecast_provenance.json", lineage)
    sources = json.loads((ROOT/"ml/artifacts/portfolio_20260926/github/sources.json").read_text())
    checks = []
    for name, expected_hash in sources["files"].items():
        path = ROOT/"ml/artifacts/portfolio_20260926/github/manticore"/name
        actual_hash = sha(path)
        assert actual_hash == expected_hash
        checks.append(dict(path=str(path.relative_to(ROOT)), sha256=actual_hash, matches_snapshot=True))
    write_json("public_solution_provenance.json", dict(snapshot=sources, file_verification=checks,
               limitations="Manticore six source files only, no final backtest keys/truth or model loaded; Trias only prior provenance entry locally; web current access failed; platform scores unverified"))
    cpu = time.process_time()-cpu_before
    assert cpu < 600
    write_json("audit_run.json", dict(protocol="M20260927 v1 M3", created_at=datetime.now(timezone.utc).isoformat(),
               script_sha256=sha(Path(__file__)), python_stdlib_only=True, cpu_seconds=cpu, wall_seconds=time.monotonic()-started,
               maxrss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, raw_read_scope="two headers only", gpu_seconds=0,
               no_training=True, preserved_history_sha256=sha(history_path)))
    print(json.dumps(dict(coverage=summary, CPU_seconds=cpu), ensure_ascii=False))


if __name__ == "__main__":
    run()
