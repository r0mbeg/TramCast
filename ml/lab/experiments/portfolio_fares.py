"""P22 preparation: ticket cohorts, retaining the exact frozen successful-event target."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from experiments.portfolio_experiment import load_history, write_json
from pipeline import aggregate, KEYS, full_grid


def cohort(values):
    text = values.fillna("").str.upper()
    education = text.str.contains("СКС|СКУ|УЧАЩ|СТУД", regex=True)
    social = text.str.contains("СКМ|СОЦ", regex=True)
    return pd.Series(np.where(education, "education", np.where(social, "social", "other")), index=values.index)


def prepare(args):
    if os.environ.get("SLURM_JOB_PARTITION") != "ais-cpu":
        raise RuntimeError("Prepare raw ticket cohorts in ais-cpu")
    if datetime.now(timezone.utc) >= datetime.fromisoformat("2026-09-27T15:40:40+00:00"):
        raise RuntimeError("Research reserve reached")
    start = time.monotonic()
    root = Path(args.output)
    root.mkdir(exist_ok=True)
    frozen = load_history(args.history)
    snapshot = json.loads(Path(args.snapshot).read_text())
    totals = {name: Counter() for name in ["education", "social", "other"]}
    tickets, sources = Counter(), {}
    for filename in ["train.csv", "test.csv"]:
        path = Path(args.raw) / filename
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        digest = digest.hexdigest()
        expected = snapshot["files"][f"dataset/{filename}"]["sha256"]
        if digest != expected:
            raise ValueError(f"Raw {filename} differs from the frozen snapshot")
        sources[filename] = digest
        count = 0
        for chunk in pd.read_csv(path, sep=";", chunksize=250000,
            usecols=["tran_date_time", "ngpt_route", "validation_result", "good_type"],
            dtype={"validation_result": "int64", "good_type": str}):
            groups = cohort(chunk.good_type)
            for name in totals:
                section = chunk.loc[groups.eq(name)]
                if section.empty:
                    continue
                _, clean, _, _ = aggregate(section, "2025-01-01", "2025-11-01")
                totals[name].update(clean.to_dict())
            tickets.update(chunk.good_type.fillna("<missing>").value_counts().to_dict())
            count += len(chunk)
            if count % 5000000 == 0:
                print(filename, count, flush=True)
        print(filename, "completed", count, flush=True)
    frame = pd.DataFrame(index=full_grid())
    for name, counts in totals.items():
        frame[name] = pd.Series(dict(counts)).reindex(frame.index, fill_value=0).astype("int64")
    result = frame.reset_index()
    result["date"] = pd.to_datetime(result.date)
    pd.testing.assert_frame_equal(result[KEYS], frozen[KEYS])
    np.testing.assert_array_equal(result[list(totals)].sum(axis=1).to_numpy(), frozen.boardings.to_numpy())
    result.to_csv(root / "hourly_fares.csv", sep=";", index=False, date_format="%Y-%m-%d")
    pd.Series(tickets, name="raw_rows").rename_axis("good_type").to_csv(root / "ticket_types.csv", sep=";")
    write_json(root / "completed.json", dict(job_id=os.environ["SLURM_JOB_ID"],
        seconds=time.monotonic()-start, sources=sources, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        history_sha256=hashlib.sha256(Path(args.history).read_bytes()).hexdigest(),
        cohort_sums={c: int(result[c].sum()) for c in totals}, exact_frozen_target_match=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    prepare(parser.parse_args())
