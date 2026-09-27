"""Verify saved audit integrity, accounting and observation masks without rereading raw."""
import argparse
import json
from pathlib import Path

import pandas as pd
from audit import sha


def verify(path):
    run = json.loads((path / "run.json").read_text())
    for name, digest in run["outputs"].items():
        if sha(path / name) != digest:
            raise ValueError(f"Changed output: {name}")
    audit = json.loads((path / "audit.json").read_text())
    frame = pd.read_csv(path / "route_hour_balance.csv", sep=";")
    assert len(frame) == 72960 and not frame.duplicated(["route", "date", "hour"]).any()
    assert (frame.boardings == frame.matched_boardings + frame.unassigned_boardings).all()
    assert frame.boardings.sum() == audit["clean_total"] == 59545140
    assert frame.matched_boardings.sum() == 0
    missing = frame.counter_status.eq("missing")
    assert missing.sum() == 730
    assert frame.loc[missing, "observed_counter"].isna().all()
    assert frame.loc[frame.counter_status.eq("structural_zero"), "observed_counter"].eq(0).all()
    assert frame.loc[frame.route.eq(5), "observed_counter"].isna().all()
    observed = frame.counter_status.eq("observed_unknown_completeness")
    assert frame.loc[observed, "observed_counter"].eq(frame.loc[observed, "boardings"]).all()
    assert ((frame.boardings == 0) & observed).sum() == 2
    pilot = pd.read_csv(path / "pilot_unassigned.csv", sep=";")
    pd.testing.assert_frame_equal(pilot, frame.loc[frame.route.eq(audit["pilot_candidate"])].reset_index(drop=True))
    assert len(pilot) == 7296
    assert audit["independently_verified_stop_accuracy"] is None
    assert next(r for r in audit["coverage"] if r["route"] == 5)["stop_match_fraction"] is None
    print("Verified hashes, 72960 unique hours, event balance, missing/zero masks, pilot and unavailable accuracy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    verify(parser.parse_args().run)
