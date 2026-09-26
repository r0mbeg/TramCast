"""Read-only archive audit. Run with pandas, numpy, lightgbm and pyarrow installed."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["route", "date", "hour"]


def temporal_features(frame):
    """Independently reproduce the archive's target-derived features for a sensitivity check."""
    frame = frame.sort_values(KEYS).copy()
    for lag in [1, 24, 168, 336, 672]:
        frame[f"lag_{lag}h"] = frame.groupby("route").boardings.shift(lag).fillna(0)
    frame = frame.sort_values(["route", "hour", "date"])
    groups = frame.groupby(["route", "hour"]).boardings
    for window in [168, 720]:
        for statistic in ["mean", "std", "max", "min"]:
            values = groups.rolling(window, min_periods=1).agg(statistic)
            frame[f"roll_{window // 24}d_{statistic}"] = values.reset_index(level=[0, 1], drop=True).fillna(0)
        frame[f"ewm_{window // 24}d"] = groups.transform(lambda x: x.ewm(span=window, min_periods=1).mean())
    for lag in [24, 168]:
        frame[f"diff_{lag}h"] = groups.diff(lag).fillna(0)
    return frame.sort_index()


def score(y, prediction, positive_only=False):
    mask = y.gt(0) if positive_only else np.ones(len(y), dtype=bool)
    return float(max(0, 1 - np.abs(y[mask] - prediction[mask]).sum() / y[mask].sum()))


def main(archive_path):
    out = ROOT / "artifacts/colleague_review"
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        def csv(name, **kwargs):
            return pd.read_csv(io.BytesIO(archive.read(name)), **kwargs)

        def parquet(name):
            return pd.read_parquet(io.BytesIO(archive.read(name)))

        frames = {s: csv(f"data/processed/{s}_features.csv") for s in ["train", "val", "forecast"]}
        submission = csv("submission.csv", sep=";")
        template = pd.read_csv(ROOT.parent / "dataset/test_submission.csv", sep=";")
        joined = frames["forecast"].merge(template, on=KEYS, validate="one_to_one")
        assert joined.boardings.eq(joined.prediction).all()
        paired = submission.merge(template, on=KEYS, suffixes=("_model", "_template"), validate="one_to_one")
        result = dict(archive_sha256=hashlib.sha256(Path(archive_path).read_bytes()).hexdigest(),
            forecast_target_equals_template_rows=len(joined),
            submission_template_correlation=float(paired.prediction_model.corr(paired.prediction_template)),
            submission_template_mae=float((paired.prediction_model - paired.prediction_template).abs().mean()),
            submission_total=int(submission.prediction.sum()),
            template_total=int(template.prediction.sum()),
            route5_nonzero=int(submission.loc[submission.route.eq(5), "prediction"].ne(0).sum()),
            route5_total=int(submission.loc[submission.route.eq(5), "prediction"].sum()),
            offhours_nonzero=int(submission.loc[submission.hour.between(1, 4), "prediction"].ne(0).sum()),
            offhours_total=int(submission.loc[submission.hour.between(1, 4), "prediction"].sum()),
            forced_zero_union_total=int(submission.loc[submission.route.eq(5) | submission.hour.between(1, 4), "prediction"].sum()))
        weather_columns = [c for c in frames["forecast"] if c.startswith("weather_")]
        result["weather_columns"] = weather_columns
        result["feature_tables"] = {}
        for name, frame in frames.items():
            rebuilt = temporal_features(frame)
            temporal = [c for c in frame if c.startswith(("lag_", "roll_", "ewm_", "diff_"))]
            np.testing.assert_allclose(frame[temporal], rebuilt[temporal], atol=1e-8)
            assert frame.groupby(["hour", "dayofweek"])[weather_columns].nunique().max().max() == 1
            result["feature_tables"][name] = dict(rows=len(frame), holiday_sum=int(frame.is_holiday.sum()),
                holiday_eve_sum=int(frame.is_holiday_eve.sum()), school_holiday_sum=int(frame.is_school_holiday.sum()),
                historical_profile_zero_fraction=float(frame.hist_median_boardings.eq(0).mean()))
        raw_weather = parquet("data/external/moscow_weather_2020_2024.parquet")
        result["raw_weather"] = dict(rows=len(raw_weather), first=str(raw_weather.datetime.min()),
            last=str(raw_weather.datetime.max()), duplicate_timestamps=int(raw_weather.datetime.duplicated().sum()),
            null_fraction=raw_weather.isna().mean().to_dict(), dtypes=raw_weather.dtypes.astype(str).to_dict())
        assert raw_weather.temp.dtype == pd.Float64Dtype()
        assert raw_weather.temp.dtype not in ["float64", "int64", "float32", "int32"]

        old_features = parquet("data/processed/val_preds_ensemble.parquet")
        non_initial = old_features.groupby("route").cumcount().ge(24)
        exact_target = old_features.lag_24h + old_features.diff_24h
        assert np.allclose(exact_target[non_initial], old_features.boardings[non_initial])
        result["old_validation"] = dict(rows=len(old_features), hours=old_features.hour.unique().tolist(),
            duplicate_keys=int(old_features.duplicated(KEYS).sum()),
            exactly_reconstructed_target_rows=int(non_initial.sum()))
        stored = parquet("data/processed/val_preds_lgbm_quantile.parquet")
        result["old_validation"].update(masked_score=score(stored.boardings, stored.lgbm_quantile_pred, True),
            full_score=score(stored.boardings, stored.lgbm_quantile_pred))
        combined = pd.concat([frames["train"], frames["val"]], ignore_index=True)
        result["final_fit"] = dict(validation_rows_used_as_training=len(frames["val"]) - 5000,
            early_stopping_route_counts=combined.tail(5000).groupby("route").size().to_dict(),
            early_stopping_start=combined.tail(5000).date.min(), early_stopping_end=combined.tail(5000).date.max())

        # Load only the text LightGBM model; never unpickle or import archive modules.
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "model.txt"
            model_path.write_bytes(archive.read("models/lgbm_quantile.txt"))
            model = lgb.Booster(model_file=str(model_path))

            def predict(frame):
                X = frame[model.feature_name()].copy()
                for c in ["route_id", "depot_id", "route_hour", "route_dow", "holiday_name"]:
                    X[c] = X[c].astype("category")
                return np.rint(np.maximum(model.predict(X, num_threads=4), 0)).astype(int)

            prediction = predict(frames["forecast"])
            regenerated = frames["forecast"][KEYS].assign(prediction=prediction).merge(
                submission, on=KEYS, suffixes=("_regenerated", "_saved"), validate="one_to_one")
            assert regenerated.prediction_regenerated.eq(regenerated.prediction_saved).all()
            result["exactly_reproduced_submission_rows"] = len(regenerated)
            known_route_hours = model.pandas_categorical[3]
            result["model_route_hour_categories"] = known_route_hours
            result["forecast_unknown_route_hour_rows"] = int((~frames["forecast"].route_hour.isin(known_route_hours)).sum())
            importance = pd.DataFrame(dict(feature=model.feature_name(), gain=model.feature_importance("gain")))
            importance["gain_fraction"] = importance.gain / importance.gain.sum()
            importance.sort_values("gain", ascending=False).to_csv(out / "feature_importance.csv", sep=";", index=False)
            result["weather_gain_fraction"] = float(importance.loc[importance.feature.str.startswith("weather_"), "gain_fraction"].sum())
            result["template_sensitivity"] = {}
            for factor in [0.5, 1.0, 2.0]:
                frame = temporal_features(frames["forecast"].assign(boardings=frames["forecast"].boardings * factor))
                values = predict(frame)
                result["template_sensitivity"][str(factor)] = dict(prediction_total=int(values.sum()),
                    ratio_to_original=float(values.sum() / prediction.sum()))
            # Diagnostic only: these still contain target leakage, not honest validation.
            result["current_feature_table_leaky_score"] = score(frames["val"].boardings, predict(frames["val"]))
        (out / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    main(parser.parse_args().archive)
