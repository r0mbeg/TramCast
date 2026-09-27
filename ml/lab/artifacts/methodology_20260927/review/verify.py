"""Independent audit. From repository root: python3 <this file> [--all-hashes].

Uses only the standard library; never imports the project's evaluator/model.
"""
import argparse
from calendar import monthrange
from collections import Counter, defaultdict
import csv
from datetime import date, timedelta
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RESULTS = OUT.parent/"route/evaluation"
ROUTES = (1, 5, 7, 11, 12, 17, 25, 26, 28, 50)
OLD = ("2025-04-30", "2025-06-30", "2025-07-31", "2025-08-31")
ORIGINS = ("2025-05-01", "2025-05-15", "2025-06-01", "2025-06-15", "2025-07-01",
           "2025-07-15", "2025-08-01", "2025-08-15", "2025-09-01")
PORT = ROOT / "ml/artifacts/portfolio_20260926"
CONT = PORT / "continuation"
ALLOWED_DOCS = {"AGENTS.md", "ml/AGENTS.md", "ml/README.md", "ml/EXPERIMENTS.md", "ml/SOLUTIONS.md"}
SOURCES = {
    "031": CONT / "adaptive_shape/study/adaptive_shape/selected",
    "029": CONT / "school_fraction/study/school_fraction/selected",
    "025": CONT / "bayes_shape_timesfm_blend/bayes_shape_timesfm_blend/selected",
    "C0": PORT / "cpu/mean_all", "C1": PORT / "gpu/daily",
}


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=";"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def key(row):
    return int(row["route"]), row["date"], int(row["hour"])


def grid(start, days):
    return {(r, (start + timedelta(days=d)).isoformat(), h)
            for r in ROUTES for d in range(days) for h in range(24)}


def published(row):
    value = float(row["prediction"])
    assert math.isfinite(value) and value >= 0, row
    r, _, h = key(row)
    assert r != 5 and h not in range(1, 5) or value == 0, row
    return math.floor(value + 0.5)


def stats(actual, predicted):
    assert len(actual) == len(predicted) and actual
    total, forecast = sum(actual), sum(predicted)
    error = sum(abs(a-p) for a, p in zip(actual, predicted))
    return dict(rows=len(actual), actual_total=total, predicted_total=forecast,
                absolute_error=error, mae=error/len(actual),
                wape=error/total if total else None,
                wape_score=max(0, 1-error/total) if total else None,
                bias=forecast-total, bias_fraction=(forecast-total)/total if total else None)


def same(got, expected):
    for name, value in got.items():
        if name not in expected:
            continue
        if expected[name] in ("", None):
            assert value is None, (name, value, "Unexpected NA in saved result")
            continue
        target = float(expected[name])
        assert value is not None and math.isclose(value, target, rel_tol=1e-12, abs_tol=1e-9), (name, value, target)


def old_temporal():
    fits, teachers = [], []
    for branch, filename in (("adaptive_shape", "daily_training.csv"),
                             ("tabular_shape", "daily_training.csv"), ("school_fraction", "training.csv")):
        for path in sorted((CONT/branch/"study/fits_42").glob("*/fit.json")):
            info = json.loads(path.read_text())
            cutoff = path.parent.name
            past = rows(path.parent/filename)
            latest = max(r["date"] for r in past)
            origins = sorted({r["origin"] for r in past})
            assert latest == info["latest_target_date"] and latest <= cutoff
            assert all((date.fromisoformat(o)+timedelta(days=61)).isoformat() <= cutoff for o in origins)
            future = rows(path.parent/"future.csv")
            assert "boardings" not in future[0] and min(r["date"] for r in future) > cutoff
            fits.append(dict(branch=branch, cutoff=cutoff, rows=len(past), latest_target=latest,
                             completed_teacher_origins=origins, future_target_absent=True))
    for path in sorted((CONT/"timesfm_teachers/daily").glob("*/source.json")):
        info = json.loads(path.read_text())
        forecast = rows(path.parent/"daily.csv")
        assert digest(path.parent/"daily.csv") == info["sha256"]
        assert info["fit_latest_date"] == info["origin"] == path.parent.name
        assert len(forecast) == 610
        assert min(r["date"] for r in forecast) == (date.fromisoformat(info["origin"])+timedelta(days=1)).isoformat()
        assert max(r["date"] for r in forecast) == info["end"]
        teachers.append(dict(origin=info["origin"], end=info["end"], rows=len(forecast), sha256=info["sha256"]))
    (OUT/"old_temporal_checks.json").write_text(json.dumps(dict(fits=fits, teachers=teachers,
        limitation="Input cutoff and teacher completion do not prove pretrained/external historical availability."), indent=2)+"\n")


def service_benchmarks():
    results = []
    for path in sorted(OUT.parent.glob("service_benchmark*.json")):
        info = json.loads(path.read_text())
        if "measurements" not in info:
            continue
        for name, expected in info["sha256"].items():
            source = ROOT/"ml"/name
            if digest(source) != expected and name == "benchmark_service.py":
                source = OUT.parent/"service_benchmark_source.py"
            assert digest(source) == expected, (path, name, "source differs")
        assert {(r["hours"], r["concurrency"]) for r in info["measurements"]} == {
            (h, c) for h in (1, 24, 720, 1464) for c in (1, 4)}
        checked = []
        for row in info["measurements"]:
            samples = row["samples"]
            assert len(samples) == row["requests"]
            assert dict(Counter(s["status"] for s in samples)) == row["statuses"]
            assert all(math.isfinite(s["seconds"]) and s["seconds"] >= 0 for s in samples)
            good = sorted(s["seconds"] for s in samples if s["status"] == "OK")
            assert math.isclose(row["success_rps"], len(good)/row["wall_seconds"], rel_tol=1e-12)
            assert math.isclose(row["completed_rps"], len(samples)/row["wall_seconds"], rel_tol=1e-12)
            for q, name in ((.5, "successful_p50_ms"), (.95, "successful_p95_ms")):
                expected = good[math.ceil(q*len(good))-1]*1000 if good else None
                assert expected == row[name]
            assert math.isclose(row["server_cpu_core_equivalents"], row["server_cpu_seconds"]/row["wall_seconds"], rel_tol=1e-12)
            checked.append(dict(hours=row["hours"], concurrency=row["concurrency"], requests=len(samples),
                                errors=len(samples)-len(good), wall_seconds=row["wall_seconds"],
                                success_rps=row["success_rps"], p95_ms=row["successful_p95_ms"]))
        results.append(dict(path=str(path.relative_to(ROOT)), sha256=digest(path), measurements=checked))
    (OUT/"service_benchmark_checks.json").write_text(json.dumps(dict(benchmarks=results,
        limitation="Microbenchmark loopback CSV service on recorded host; no deployment quota or sustained production capacity claim."), indent=2)+"\n")
    return len(results)


def m1_check(history, truth):
    folder = OUT.parent/"route/m1"
    if not (folder/"run_started.json").exists():
        return dict(status="pending", reason="No run metadata")
    run = json.loads((folder/"run_started.json").read_text())
    assert run["origins"] == list(ORIGINS) and run["horizons"] == [1, 7, 30, 61] and run["trials"] == 0
    assert run["model_weights"] == {"model.safetensors": "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"}
    assert (run["seed"], run["quantile"], run["context"], run["dtype"], run["batch_size"], run["cross_learning"]) == (42, .5, 512, "float32", 32, False)
    cache, checked, missing = {}, [], []
    for method in ("C0", "C1"):
        for mode in ("frozen", "weekly"):
            for origin in ORIGINS:
                target = folder/method/mode/origin
                if not (target/"forecast.csv").exists():
                    missing.append(dict(method=method, mode=mode, origin=origin)); continue
                start = date.fromisoformat(origin)
                offsets = (0,) if mode == "frozen" else range(0, 61, 7)
                incomplete = [(start+timedelta(days=offset-1)).isoformat() for offset in offsets
                              if not all((target/(start+timedelta(days=offset-1)).isoformat()/name).exists()
                                         for name in ("raw.csv", "published.csv", "trace.json"))]
                if incomplete:
                    missing.append(dict(method=method, mode=mode, origin=origin, reason="Incomplete local block transfer", cutoffs=incomplete)); continue
                whole = rows(target/"forecast.csv")
                final = {key(r): int(r["prediction"]) for r in whole}
                assert len(final) == len(whole) == 14640 and set(final) == grid(start, 61)
                stitched = {}
                for offset in offsets:
                    cutoff = start + timedelta(days=offset-1)
                    emitted = start+timedelta(days=min(60, offset+6)) if mode == "weekly" else start+timedelta(days=60)
                    segment = target/cutoff.isoformat()
                    trace = json.loads((segment/"trace.json").read_text())
                    assert trace["cutoff"] == cutoff.isoformat() and trace["end"] == (cutoff+timedelta(days=61)).isoformat()
                    assert trace["emitted_end"] == emitted.isoformat() and trace["inference_horizon"] == 61
                    assert digest(segment/"raw.csv") == trace["raw_sha256"] and digest(segment/"published.csv") == trace["published_sha256"]
                    if cutoff not in cache:
                        past = [r for r in history if r["date"] <= cutoff.isoformat()]
                        stream = io.StringIO(newline="")
                        writer = csv.writer(stream, lineterminator="\n")
                        writer.writerow(("route", "date", "hour", "boardings"))
                        writer.writerows((r["route"], r["date"], r["hour"], r["boardings"]) for r in past)
                        sums, counts = defaultdict(int), Counter()
                        for r in past:
                            group = (int(r["route"]), date.fromisoformat(r["date"]).weekday(), int(r["hour"]))
                            sums[group] += int(r["boardings"]); counts[group] += 1
                        cache[cutoff] = (len(past), max(r["date"] for r in past), hashlib.sha256(stream.getvalue().encode()).hexdigest(),
                                         {k: value/counts[k] for k, value in sums.items()})
                    size, latest, input_hash, means = cache[cutoff]
                    assert trace["input_rows"] == size and trace["input_max"] == latest and trace["input_sha256"] == input_hash
                    raw = rows(segment/"raw.csv")
                    values = {key(r): published(r) for r in raw}
                    saved = rows(segment/"published.csv")
                    assert len(raw) == len(values) == 14640 and set(values) == grid(cutoff+timedelta(days=1), 61)
                    assert values == {key(r): int(r["prediction"]) for r in saved} and len(saved) == 14640
                    if method == "C0":
                        for r in raw:
                            route, day, hour = key(r)
                            assert math.isclose(float(r["prediction"]), means[route, date.fromisoformat(day).weekday(), hour], rel_tol=1e-12, abs_tol=1e-9)
                    stitched.update({k: v for k, v in values.items() if k[1] <= emitted.isoformat()})
                assert stitched == final
                for r in whole:
                    lead = (date.fromisoformat(r["date"])-date.fromisoformat(r["prediction_cutoff"])).days
                    assert lead == int(r["actual_lead"]) and 1 <= lead <= (7 if mode == "weekly" else 61)
                metrics = []
                for horizon in (1, 7, 30, 61):
                    chosen = [k for k in sorted(final) if (date.fromisoformat(k[1])-start).days < horizon]
                    for level in ("hourly", "daily", "route_calendar_month"):
                        a, p = defaultdict(int), defaultdict(int)
                        for k in chosen:
                            group = k if level == "hourly" else k[:2] if level == "daily" else (k[0], k[1][:7])
                            a[group] += truth[k]; p[group] += final[k]
                        metric = stats(list(a.values()), [p[k] for k in a])
                        metric.update(method=method, mode=mode, origin=origin, horizon=horizon, level=level,
                                      n=metric["rows"], bias_ratio=metric["bias_fraction"])
                        metrics.append(metric)
                checked.extend(metrics)
    metrics_path = RESULTS/"metrics.csv"
    if metrics_path.exists():
        saved = {(r["method"], r["mode"], r["origin"], int(r["horizon"]), r["level"]): r for r in rows(metrics_path)}
        for metric in checked:
            index = tuple(metric[k] for k in ("method", "mode", "origin", "horizon", "level"))
            same({k: v for k, v in metric.items() if isinstance(v, (int, float)) or v is None}, saved[index])
    output = dict(status="passed" if not missing and metrics_path.exists() else "partial", checked_metric_rows=len(checked),
                  missing_forecasts=missing, input_cutoffs_verified=len(cache), metrics=checked)
    (OUT/"m1_checks.json").write_text(json.dumps(output, indent=2)+"\n")
    return {k: v for k, v in output.items() if k != "metrics"}


def m3_check(history, truth):
    folder = OUT.parent/"data"
    if not (folder/"coverage_summary.json").exists():
        return dict(status="pending")
    def coverage(group):
        active = [r for r in group if int(r["route"]) != 5 and int(r["hour"]) not in range(1, 5)]
        events = sum(int(r["working_events"]) > 0 for r in active)
        successes = sum(int(r["boardings"]) > 0 for r in active)
        return dict(cells=len(group), structural_route5=sum(int(r["route"]) == 5 for r in group),
                    structural_night_non5=sum(int(r["route"]) != 5 and 1 <= int(r["hour"]) <= 4 for r in group),
                    active_cells=len(active), active_event_present=events, active_success_present=successes,
                    active_no_events=len(active)-events, active_no_success=len(active)-successes,
                    active_unsuccessful_only=events-successes,
                    supplied_label_present=sum(r["supplied_label_present"] == "True" for r in group),
                    raw_total=sum(int(r["raw_boardings"]) for r in group), clean_total=sum(int(r["boardings"]) for r in group),
                    working_events=sum(int(r["working_events"]) for r in group))
    summary = json.loads((folder/"coverage_summary.json").read_text())
    same(coverage(history), summary)
    assert summary["complete_counter_cells"] is None
    for filename, columns in (("coverage_by_route.csv", ("route",)),
                              ("coverage_by_route_hour.csv", ("route", "hour")),
                              ("coverage_by_route_date.csv", ("route", "date"))):
        grouped = defaultdict(list)
        for row in history:
            grouped[tuple(row[c] for c in columns)].append(row)
        saved = {tuple(r[c] for c in columns): r for r in rows(folder/filename)}
        assert set(grouped) == set(saved)
        for group, value in grouped.items():
            same(coverage(value), saved[group])
    labels = rows(ROOT/"dataset/labels/labels_day_train.csv") + rows(ROOT/"dataset/labels/labels_day_test.csv")
    original = {key(r): int(r["boardings"]) for r in labels}
    assert len(labels) == len(original) == 57551
    for row in history:
        k = key(row)
        assert int(row["raw_boardings"]) == original.get(k, 0)
        assert (row["supplied_label_present"] == "True") == (k in original)
        assert (row["working_events_observed"] == "True") == (int(row["working_events"]) > 0)
        assert (row["successful_working_observed"] == "True") == (int(row["boardings"]) > 0)
        assert int(row["raw_boardings"])-int(row["boardings"]) == int(row["removed_boardings"])
    missing = rows(folder/"active_no_success_cells.csv")
    absent = {key(r) for r in history if int(r["route"]) != 5 and int(r["hour"]) not in range(1, 5) and not int(r["boardings"])}
    assert len(missing) == 732 and {key(r) for r in missing} == absent
    masks = rows(folder/"sensitivity_historical.csv")
    actual_rows = {key(r): r for r in history}
    cache = {}
    for row in masks:
        method, cutoff, mask = row["method"], row["cutoff"], row["mask"]
        if (method, cutoff) not in cache:
            raw = rows(SOURCES[method]/f"raw_{cutoff}.csv")
            cache[method, cutoff] = {key(r): published(r) for r in raw}
        predictions = cache[method, cutoff]
        selected = []
        for k in sorted(predictions):
            active = k[0] != 5 and k[2] not in range(1, 5)
            events = int(actual_rows[k]["working_events"]) > 0
            success = truth[k] > 0
            if (mask == "full_absent_as_zero" or mask == "legacy_observed_including_structural" and (not active or events)
                or mask == "working_any_event_present" and active and events
                or mask == "working_success_present" and active and success):
                selected.append(k)
        metric = stats([truth[k] for k in selected], [predictions[k] for k in selected])
        metric.update(MAE=metric["mae"], WAPE=metric["wape"], score=metric["wape_score"], signed_bias=metric["bias"])
        same(metric, row)
    assert len(masks) == 48 and len(cache) == 12
    output = dict(status="passed", historical_keys_verified=72960, labels_verified=57551,
                  sensitivity_rows_verified=len(masks), coverage_tables_verified=3,
                  active_no_success=732, active_no_events=730, complete_counter_cells=None,
                  limitation="Independent aggregate checks and raw headers; raw events not rescanned, no stop-time truth exists.")
    (OUT/"m3_checks.json").write_text(json.dumps(output, indent=2)+"\n")
    return output


def m2_check(history, truth):
    folder = OUT.parent/"route/m2"
    if not (folder/"completed.json").exists():
        return dict(status="pending")
    run = json.loads((folder/"run_started.json").read_text())
    assert run["params"] == dict(correction_days=14, shape_mix=.5, route_season=False) and run["trials"] == 0
    assert run["events"] == dict(july="2025-07-09", autumn="2025-09-05")
    forecasts, checked = {}, []
    saved_path = RESULTS/"metrics.csv"
    saved = {(r["method"], r["mode"], r["origin"], int(r["horizon"]), r["level"]): r for r in rows(saved_path)} if saved_path.exists() else {}
    for mode in ("no_notices", "published_by_cutoff", "all_hindsight"):
        for origin in ORIGINS:
            target = folder/mode/origin
            start = date.fromisoformat(origin); cutoff = start-timedelta(days=1)
            trace = json.loads((target/"trace.json").read_text())
            assert trace["cutoff"] == cutoff.isoformat() and trace["end"] == (cutoff+timedelta(days=61)).isoformat()
            expected_events = [] if mode == "no_notices" else list(run["events"]) if mode == "all_hindsight" else [e for e, d in run["events"].items() if date.fromisoformat(d) <= cutoff]
            assert trace["available_events"] == expected_events and trace["parameter_selection_hindsight"]
            raw = rows(target/"raw.csv"); values = {key(r): published(r) for r in raw}
            published_rows = rows(target/"published.csv")
            assert len(raw) == len(values) == len(published_rows) == 14640 and set(values) == grid(start, 61)
            assert values == {key(r): int(r["prediction"]) for r in published_rows}
            assert digest(target/"raw.csv") == trace["raw_sha256"] and digest(target/"published.csv") == trace["published_sha256"]
            past = [r for r in history if r["date"] <= cutoff.isoformat()]
            stream = io.StringIO(newline=""); writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("route", "date", "hour", "boardings"))
            writer.writerows((r["route"], r["date"], r["hour"], r["boardings"]) for r in past)
            assert trace["input_rows"] == len(past) and trace["input_max"] == max(r["date"] for r in past)
            assert hashlib.sha256(stream.getvalue().encode()).hexdigest() == trace["input_sha256"]
            forecasts[mode, origin] = values
            for horizon in (1, 7, 30, 61):
                chosen = [k for k in sorted(values) if (date.fromisoformat(k[1])-start).days < horizon]
                for level in ("hourly", "daily", "route_calendar_month"):
                    a, p = defaultdict(int), defaultdict(int)
                    for k in chosen:
                        group = k if level == "hourly" else k[:2] if level == "daily" else (k[0], k[1][:7])
                        a[group] += truth[k]; p[group] += values[k]
                    metric = stats(list(a.values()), [p[k] for k in a])
                    metric.update(method="P10/P11", mode=mode, origin=origin, horizon=horizon, level=level,
                                  n=metric["rows"], bias_ratio=metric["bias_fraction"])
                    if saved:
                        same({k: v for k, v in metric.items() if isinstance(v, (int, float)) or v is None}, saved["P10/P11", mode, origin, horizon, level])
                    checked.append(metric)
    for origin in ORIGINS:
        if date.fromisoformat(origin)-timedelta(days=1) < date(2025, 7, 9):
            assert forecasts["no_notices", origin] == forecasts["published_by_cutoff", origin]
    output = dict(status="passed" if saved else "traces passed; metrics pending", checked_metric_rows=len(checked),
                  forecasts=27, complete_origins=9, metrics=checked,
                  limitation="Notice cutoff is verified against declared publication sources; all arms share globally hindsight selected parameters.")
    (OUT/"m2_checks.json").write_text(json.dumps(output, indent=2)+"\n")
    return {k: v for k, v in output.items() if k != "metrics"}


def summary_check():
    folder = RESULTS
    if not (folder/"summary.csv").exists():
        return dict(status="pending")
    metrics = rows(folder/"metrics.csv")
    grouped = defaultdict(list)
    for row in metrics:
        grouped[tuple(row[k] for k in ("method", "mode", "horizon", "level"))].append(row)
    checked = 0
    for row in rows(folder/"summary.csv"):
        group = grouped[tuple(row[k] for k in ("method", "mode", "horizon", "level"))]
        if row["scope"] == "shared_existing5":
            group = [r for r in group if r["origin"] in ("2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01")]
        else:
            assert row["scope"] == "all_planned9"
        present = [r for r in group if r["actual_total"]]
        assert len(group) == int(row["expected_origins"]) and len(present) == int(row["available_origins"])
        complete = len(group) == len(present)
        assert (row["complete"] == "True") == complete
        if complete:
            scores = [float(r["wape_score"]) for r in present if r["wape_score"]]
            total = sum(float(r["actual_total"]) for r in present)
            wape = sum(float(r["absolute_error"]) for r in present)/total if total else None
            expected = dict(mean_score=sum(scores)/len(scores), minimum_score=min(scores), maximum_score=max(scores),
                            score_range=max(scores)-min(scores), pooled_wape=wape, pooled_score=max(0, 1-wape) if wape is not None else None)
            same(expected, row)
        else:
            assert all(row[k] == "" for k in ("mean_score", "minimum_score", "maximum_score", "score_range", "pooled_wape", "pooled_score")), row
            assert all(r.get("reason") for r in group if not r["actual_total"])
        checked += 1
    return dict(status="passed", summaries_verified=checked, limitation="Pooled values repeat overlapping target keys; no independent-sample interpretation.")


def breakdown_check(truth):
    folder = OUT.parent/"route"
    if not (RESULTS/"breakdown.csv").exists():
        return dict(status="pending")
    forecasts, groups = {}, {}
    for path in (folder/"m1").glob("*/*/*/forecast.csv"):
        method, mode, origin = path.parts[-4:-1]
        forecasts[method, mode, origin] = {key(r): int(r["prediction"]) for r in rows(path)}
    for path in (folder/"m2").glob("*/*/published.csv"):
        mode, origin = path.parts[-3:-1]
        forecasts["P10/P11", mode, origin] = {key(r): int(r["prediction"]) for r in rows(path)}
    for path in (RESULTS/"retrospective").glob("*/*/published.csv"):
        method, origin = path.parts[-3:-1]
        forecasts[method, "retrospective", origin] = {key(r): int(r["prediction"]) for r in rows(path)}
    checked = 0
    for row in rows(RESULTS/"breakdown.csv"):
        identity = row["method"], row["mode"], row["origin"]
        if identity not in groups:
            values = forecasts[identity]; origin = date.fromisoformat(row["origin"])
            buckets = defaultdict(list)
            for k in sorted(values):
                route, day, hour = k; stamp = date.fromisoformat(day); lead = (stamp-origin).days+1
                weekend = stamp.weekday() >= 5
                movement = (route in (7, 50) and "2025-07-10" <= day <= "2025-08-10"
                            or route == 7 and weekend and "2025-08-16" <= day <= "2025-09-05"
                            or route in (7, 50) and weekend and "2025-09-06" <= day <= "2025-11-14")
                transition = "2025-05-18" <= day <= "2025-06-14" or "2025-08-18" <= day <= "2025-09-14"
                regime = "movement" if movement else "transition" if transition else "stable"
                degradation = "1-7" if lead <= 7 else "8-30" if lead <= 30 else "31-61"
                for dimension, value in (("route", str(route)), ("month", day[:7]), ("degradation", degradation), ("regime", regime)):
                    buckets[dimension, value].append((k, lead, truth[k], values[k]))
            groups[identity] = buckets
        chosen = [r for r in groups[identity][row["dimension"], row["value"]] if r[1] <= int(row["horizon"])]
        actual, prediction = defaultdict(int), defaultdict(int)
        for k, _, a, p in chosen:
            group = k if row["level"] == "hourly" else k[:2] if row["level"] == "daily" else (k[0], k[1][:7])
            actual[group] += a; prediction[group] += p
        result = stats(list(actual.values()), [prediction[k] for k in actual])
        result.update(n=result["rows"], bias_ratio=result["bias_fraction"])
        same(result, row)
        first = date.fromisoformat(min(r[0][1] for r in chosen))
        last = date.fromisoformat(max(r[0][1] for r in chosen))
        partial = first.day != 1 or last.day != monthrange(last.year, last.month)[1]
        assert (row["partial_month"] == "True") == partial
        checked += 1
    return dict(status="passed", breakdown_rows_verified=checked, dimensions=["route", "month", "degradation", "regime"],
                limitation="Regime masks are descriptive retrospective labels; no causal repair effect inferred.")


def exported_aggregates(truth):
    if not (RESULTS/"daily_aggregates.csv").exists():
        return dict(status="pending")
    folder = OUT.parent/"route"
    paths = []
    for path in (folder/"m1").glob("*/*/*/forecast.csv"):
        paths.append((path, *path.parts[-4:-1]))
    for path in (folder/"m2").glob("*/*/published.csv"):
        mode, origin = path.parts[-3:-1]; paths.append((path, "P10/P11", mode, origin))
    for path in (RESULTS/"retrospective").glob("*/*/published.csv"):
        method, origin = path.parts[-3:-1]; paths.append((path, method, "retrospective", origin))
    saved_metrics = {(r["method"], r["mode"], r["origin"], int(r["horizon"]), r["level"]): r for r in rows(RESULTS/"metrics.csv")}
    daily, monthly = {}, {}
    retrospective_metrics = 0
    for path, method, mode, origin in paths:
        values = {key(r): int(r["prediction"]) for r in rows(path)}
        start = date.fromisoformat(origin)
        assert len(values) == 14640 and set(values) == grid(start, 61)
        if mode == "retrospective":
            lineage = json.loads((path.parent/"lineage.json").read_text())
            source = ROOT/"ml"/lineage["source"]
            assert digest(source) == lineage["sha256"]
            raw = rows(source)
            assert values == {key(r): published(r) for r in raw} and len(raw) == 14640
        for horizon in (1, 7, 30, 61):
            a, p, days = defaultdict(int), defaultdict(int), defaultdict(set)
            chosen = [k for k in sorted(values) if (date.fromisoformat(k[1])-start).days < horizon]
            for k in chosen:
                route, day, hour = k
                day_key = method, mode, origin, horizon, route, day
                month_key = method, mode, origin, horizon, route, day[:7]
                for group in (day_key, month_key):
                    a[group] += truth[k]; p[group] += values[k]
                days[month_key].add(day)
            daily.update({k: (v, p[k]) for k, v in a.items() if len(k[-1]) == 10})
            monthly.update({k: (v, p[k], len(days[k])) for k, v in a.items() if len(k[-1]) == 7})
            if mode == "retrospective":
                for level in ("hourly", "daily", "route_calendar_month"):
                    aa, pp = defaultdict(int), defaultdict(int)
                    for k in chosen:
                        group = k if level == "hourly" else k[:2] if level == "daily" else (k[0], k[1][:7])
                        aa[group] += truth[k]; pp[group] += values[k]
                    metric = stats(list(aa.values()), [pp[k] for k in aa]); metric.update(n=metric["rows"], bias_ratio=metric["bias_fraction"])
                    same(metric, saved_metrics[method, mode, origin, horizon, level]); retrospective_metrics += 1
    saved_day = rows(RESULTS/"daily_aggregates.csv")
    assert len(saved_day) == len(daily) == 79200
    for row in saved_day:
        k = row["method"], row["mode"], row["origin"], int(row["horizon"]), int(row["route"]), row["date"]
        assert daily[k] == (int(row["boardings"]), int(row["prediction"]))
    saved_month = rows(RESULTS/"route_month_aggregates.csv")
    assert len(saved_month) == len(monthly) == 4600
    for row in saved_month:
        k = row["method"], row["mode"], row["origin"], int(row["horizon"]), int(row["route"]), row["month"]
        assert monthly[k] == (int(row["boardings"]), int(row["prediction"]), int(row["included_days"]))
        year, month = map(int, row["month"].split("-"))
        assert int(row["calendar_days"]) == monthrange(year, month)[1]
        assert (row["partial_month"] == "True") == (int(row["included_days"]) != int(row["calendar_days"]))
    return dict(status="passed", daily_rows_verified=len(daily), route_month_rows_verified=len(monthly),
                retrospective_metric_rows_verified=retrospective_metrics, final_integer_hours=True)


def source_provenance():
    folder = OUT.parent/"route"
    run = json.loads((folder/"m1/run_started.json").read_text())
    for name, expected in run["sha256"].items():
        source = folder/"executed_M1.py" if name.startswith("/beegfs/") else ROOT/"ml"/name
        assert digest(source) == expected
    dependencies = json.loads((folder/"dependency_match.json").read_text())
    assert len(dependencies) == 8
    for row in dependencies:
        assert row["exact"] and digest(ROOT/row["local_path"]) == row["sha256"]
    assert (folder/"model_config_provenance.txt").read_text().splitlines()[:3] == [
        "ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031  model/config.json",
        "29ec3766d36d6f73f0696f85560a422f50e8498c", "f3207394965100a2aa18820b87da3d2bbad3b1f0"]
    assert (folder/"model_weights_metadata.txt").read_text().splitlines()[:2] == [
        "29ec3766d36d6f73f0696f85560a422f50e8498c", "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"]
    evaluator = json.loads((RESULTS/"evaluator_run.json").read_text())
    assert digest(folder/"executed_evaluator.py") == evaluator["code_sha256"]
    return dict(status="passed", M1_sources_verified=len(run["sha256"]), M2_dependencies_verified=8,
                model_metadata_matches_prior=True, evaluator_snapshot_verified=True,
                limitation="Downloaded metadata corroborates revision; pretrained corpus overlap and historical release date of this precise revision not audited.")


def run(all_hashes=False):
    assert stats([10, 20], [8, 22])["wape_score"] == 1-4/30
    assert stats([0], [0])["wape"] is None
    old_temporal()
    benchmark_count = service_benchmarks()
    manifest = json.loads((OUT.parent / "protected_manifest.json").read_text())["files"]
    selected = manifest if all_hashes else {p: h for p, h in manifest.items() if
        p.startswith("ml/submissions/") or p in ("ml/submission.csv", "ml/service.py", "ml/forecast_bundle.json", "ml/artifacts/hourly_clean.csv", "ml/pipeline.py")}
    changed = [p for p, h in selected.items() if not (ROOT/p).is_file() or digest(ROOT/p) != h]
    history = rows(ROOT / "ml/artifacts/hourly_clean.csv")
    truth = {key(r): int(r["boardings"]) for r in history}
    assert len(truth) == len(history) == 72960
    assert set(truth) == grid(date(2025, 1, 1), 304)
    assert sum(truth.values()) == 59545140
    assert all(v == 0 for (r, _, h), v in truth.items() if r == 5 or 1 <= h <= 4)
    m1 = m1_check(history, truth)
    m3 = m3_check(history, truth)
    m2 = m2_check(history, truth)
    summaries = summary_check()
    breakdown = breakdown_check(truth)
    aggregates = exported_aggregates(truth)
    provenance = source_provenance()
    result = []
    for method, folder in SOURCES.items():
        saved_metrics = {r["cutoff"]: r for r in rows(folder / "metrics.csv")}
        saved_predictions = rows(folder / "predictions.csv")
        for cutoff in OLD:
            raw = rows(folder / f"raw_{cutoff}.csv")
            start = date.fromisoformat(cutoff) + timedelta(days=1)
            expected = grid(start, 61)
            prediction = {key(r): published(r) for r in raw}
            assert len(raw) == len(prediction) == 14640 and set(prediction) == expected
            saved = {key(r): int(r["prediction"]) for r in saved_predictions if r["cutoff"] == cutoff}
            assert prediction == saved, (method, cutoff, "raw publication mismatch")
            ordered = sorted(expected)
            metric = stats([truth[k] for k in ordered], [prediction[k] for k in ordered])
            same(metric, saved_metrics[cutoff])
            result.append(dict(method=method, cutoff=cutoff, raw_sha256=digest(folder/f"raw_{cutoff}.csv"), **metric))
    bundle = json.loads((ROOT/"ml/forecast_bundle.json").read_text())
    source = ROOT / "ml" / bundle["prediction_file"]
    assert digest(source) == bundle["prediction_sha256"]
    submission = rows(source)
    assert len(submission) == 14640 and {key(r) for r in submission} == grid(date(2025, 11, 1), 61)
    assert all(int(r["prediction"]) == published(r) for r in submission)
    output = dict(independent_evaluator="stdlib formulas; no project imports", protected_checked=len(selected),
                  protected_changed=changed, old_window_comparisons=result, service_csv_sha256=digest(source),
                  service_benchmarks_checked=benchmark_count, m1=m1, m2=m2, m3=m3, summaries=summaries,
                  breakdown=breakdown, aggregates=aggregates, provenance=provenance,
                  authorized_documentation_changes=[p for p in changed if p in ALLOWED_DOCS],
                  unauthorized_protected_changes=[p for p in changed if p not in ALLOWED_DOCS])
    (OUT/"verification.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(dict(protected_checked=len(selected), protected_changed=changed,
                         comparisons=len(result), service_grid_points=len(submission)), indent=2))
    assert not output["unauthorized_protected_changes"], changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-hashes", action="store_true")
    run(parser.parse_args().all_hashes)
