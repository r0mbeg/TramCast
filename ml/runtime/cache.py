"""One atomic, persistent result per immutable recipe and data snapshot."""
import csv
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import time

from bundle import Bundle

LOG = logging.getLogger("tramcast.ml")


def cache_key(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class ForecastCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS forecasts "
                       "(key TEXT PRIMARY KEY, csv BLOB NOT NULL, sha256 TEXT NOT NULL)")

    @staticmethod
    def read(db, key, metadata):
        row = db.execute("SELECT csv, sha256 FROM forecasts WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        try:
            return Bundle.from_csv(dict(metadata, prediction_sha256=row[1]), row[0])
        except (ValueError, TypeError, KeyError, OverflowError, csv.Error):
            LOG.warning("Corrupt cache entry will be recomputed: %s", key)
            return None

    def get(self, metadata, compute, check_active):
        """compute returns CSV bytes; check_active raises on cancellation/deadline."""
        key = cache_key(metadata)
        db = sqlite3.connect(self.path, timeout=0, isolation_level=None)
        try:
            while True:
                check_active()
                cached = self.read(db, key, metadata)
                if cached is not None:
                    LOG.info(json.dumps(dict(event="cache_hit", key=key)))
                    return cached
                try:
                    # ponytail: one cold calculation per local cache database. Move
                    # scheduling to the Go worker pool if parallel inference is needed.
                    db.execute("BEGIN IMMEDIATE")
                    break
                except sqlite3.OperationalError as error:
                    if error.sqlite_errorcode not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                        raise
                    time.sleep(0.05)
            # Another process may have committed between SELECT and taking the lock.
            cached = self.read(db, key, metadata)
            if cached is not None:
                return cached
            check_active()
            started = time.monotonic()
            content = compute(check_active)
            digest = hashlib.sha256(content).hexdigest()
            bundle = Bundle.from_csv(dict(metadata, prediction_sha256=digest), content)
            check_active()
            db.execute("INSERT OR REPLACE INTO forecasts VALUES (?, ?, ?)", (key, content, digest))
            db.commit()
            LOG.info(json.dumps(dict(event="computed", key=key, seconds=time.monotonic()-started)))
            return bundle
        finally:
            # Error, cancellation or worker death never publishes a partial result.
            if db.in_transaction:
                db.rollback()
            db.close()
