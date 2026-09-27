"""Fixed 030 dependencies, reusing lab functions in the builder's isolated workspace."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build import ROOT, digest, write_json

CUTOFF, END = "2025-10-31", "2025-12-31"
TIMESFM_CONFIG = dict(max_context=512, max_horizon=128, normalize_inputs=True,
                     window_size=0, per_core_batch_size=8, use_continuous_quantile_head=True,
                     force_flip_invariance=True, infer_is_positive=True,
                     fix_quantile_crossing=True, return_backcast=False)


def params(name):
    return json.loads((ROOT / f"{name}/{name}/selection.json").read_text())["params"]


def save(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep=";", index=False, date_format="%Y-%m-%d")


def cached_quantiles(folder, cutoff, end, matrix):
    from experiments.portfolio_timesfm import WEIGHTS_SHA256, WHEEL_SHA256
    import hashlib
    path = folder / f"normal_ratio_{cutoff}_july_verified.npz"
    if not path.exists():
        return None
    metadata = json.loads(path.with_suffix(".json").read_text())
    if digest(path) != metadata["sha256"]:
        raise ValueError("Corrupt TimesFM quantiles cache")
    required = dict(cutoff=cutoff, end=end, representation="normal_ratio", july_verified=True,
                    model_sha256=WEIGHTS_SHA256, wheel_sha256=WHEEL_SHA256,
                    input_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(), config=TIMESFM_CONFIG)
    if any(metadata.get(key) != value for key, value in required.items()):
        return None
    with np.load(path, allow_pickle=False) as arrays:
        if "inputs" in arrays and not np.array_equal(arrays["inputs"], matrix):
            raise ValueError("TimesFM cache input matrix differs")
        values = arrays["quantiles"].copy()
    if values.shape != (9, 61, 10) or not np.isfinite(values).all():
        raise ValueError("Invalid TimesFM cached quantiles")
    return values


def teachers(history, weights, cache, threads):
    from experiments.portfolio_timesfm import series_inputs, teacher_daily, WEIGHTS_SHA256
    from experiments.portfolio_windows import inner_windows
    model = None
    evidence = []
    for cutoff, end in inner_windows(CUTOFF) + [(CUTOFF, END)]:
        matrix, restoration = series_inputs(history, cutoff, end, "normal_ratio", july_verified=True)
        values = cached_quantiles(cache, cutoff, end, matrix) if cache else None
        reused = values is not None
        if values is None:
            if weights is None:
                raise ValueError(f"New TimesFM inputs at {cutoff}; provide --timesfm-model, stale teachers cannot be reused")
            if model is None:
                from importlib.metadata import version
                import torch
                import timesfm
                if digest(weights) != WEIGHTS_SHA256:
                    raise ValueError("Unverified TimesFM weights")
                if version("timesfm") != "2.0.2" or not torch.cuda.is_available():
                    raise RuntimeError("Teacher rebuild requires timesfm 2.0.2 and NVIDIA CUDA")
                torch.set_num_threads(threads)
                torch.set_num_interop_threads(1)
                torch.manual_seed(42)
                model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
                model.load_checkpoint(weights, torch_compile=False)
                model.compile(timesfm.ForecastConfig(**TIMESFM_CONFIG))
            import torch
            with torch.inference_mode():
                _, values = model.forecast(horizon=61, inputs=[row.copy() for row in matrix])
        if values.shape != (9, 61, 10) or not np.isfinite(values).all():
            raise ValueError("Invalid TimesFM output")
        folder = ROOT / "timesfm_teachers/daily" / cutoff
        path = folder / "daily.csv"
        save(teacher_daily(history, cutoff, end, values, restoration), path)
        # Retain actual model inputs/output, including reused quantiles, for independent audit.
        np.savez_compressed(folder / "quantiles.npz", inputs=matrix, restoration=restoration, quantiles=values)
        info = dict(origin=cutoff, end=end, fit_latest_date=cutoff, quantile_index=3,
                    july_verified=True, model_sha256=WEIGHTS_SHA256, sha256=digest(path),
                    quantiles_sha256=digest(folder / "quantiles.npz"), reused_quantiles=reused,
                    config=TIMESFM_CONFIG)
        if reused:
            source = cache / f"normal_ratio_{cutoff}_july_verified.npz"
            info["cached_quantiles_sha256"] = digest(source)
            info["cached_metadata_sha256"] = digest(source.with_suffix(".json"))
        else:
            from importlib.metadata import version
            info["libraries"] = {name: version(name) for name in ("torch", "timesfm", "numpy")}
        write_json(folder / "source.json", info)
        evidence.append(info)
        print(f"  TimesFM {cutoff}: {'verified cache' if reused else 'computed'}", flush=True)
    if model is not None:
        del model
        import torch
        torch.cuda.empty_cache()
    return evidence


def parents(history, weather):
    from experiments.portfolio_bayes_volume import bayes_volume_forecast
    from experiments.portfolio_direct_daily import direct_daily_forecast
    from experiments.portfolio_conditional_shape import shape_forecast
    from experiments.portfolio_bayes_direct import forecast as forecast_021
    from experiments.portfolio_bayes_shape import forecast as forecast_024, INNER
    from experiments.portfolio_combine import mix, raw_frame
    from experiments.portfolio_windows import inner_windows
    from experiments.portfolio_school_fraction import fit
    from experiments.portfolio_timesfm_errors import examples
    from experiments.portfolio_route_allocation import preserve_network

    # 024's completed error examples use the original (pre-restoration-fix) P29 teachers.
    # Rebuilding them with verified-July operations would silently change the 030 recipe.
    volume = bayes_volume_forecast(history, CUTOFF, END, params("bayes_volume"))
    direct = direct_daily_forecast(history, CUTOFF, END, params("direct_daily"))
    blend = mix(volume, direct, .75)  # Selected P28: 75% Bayesian + 25% direct.
    conditional = shape_forecast(history, CUTOFF, params("conditional_shape"), weather, blend)
    save(conditional, ROOT / f"conditional_shape/conditional_shape/selected/raw_{CUTOFF}.csv")
    for cutoff, end in inner_windows(CUTOFF):
        base = raw_frame(ROOT / f"bayes_volume/inner/{cutoff}/raw_{cutoff}.csv", cutoff, end)
        shape = shape_forecast(history, cutoff, params("conditional_shape"), weather, base)
        path = INNER / cutoff / f"raw_{cutoff}.csv"
        save(shape, path)
        write_json(path.parent / "source.json", dict(origin=cutoff, end=end, fit_latest_date=cutoff,
                   params=params("conditional_shape"), sha256=digest(path)))
        print(f"  Hourly training forecast {cutoff}", flush=True)

    verified = ROOT / "verified_july"
    verified.mkdir(parents=True, exist_ok=True)
    raw021 = forecast_021(history, CUTOFF, END,
                         dict(july_verified=True, season="route", strength=1., uncertainty=1, history_days=224), verified)
    save(raw021, verified / f"verified_july/selected/raw_{CUTOFF}.csv")
    raw024 = forecast_024(history, CUTOFF, END, "half", ROOT / "bayes_shape/study")
    save(raw024, ROOT / f"bayes_shape/study/bayes_shape/selected/raw_{CUTOFF}.csv")

    past, future = examples(history, CUTOFF, END)
    shares = fit(past, future, CUTOFF, ROOT / f"school_fraction/fits/{CUTOFF}", 42)
    learned = raw024.merge(shares, on=["route", "date"], validate="many_to_one")
    volume = raw024.groupby(["route", "date"]).prediction.transform("sum")
    learned["prediction"] = raw024.prediction.div(volume.where(volume.gt(0))).fillna(0) * learned.estimated_share
    raw029 = mix(preserve_network(raw024, learned.drop(columns="estimated_share")), raw024, .5)
    save(raw029, ROOT / f"school_fraction/study/school_fraction/selected/raw_{CUTOFF}.csv")
    return raw029


def package_inputs(history, base, destination):
    from experiments.portfolio_tabular_shape import training, basis, SOURCE, HOURS
    from experiments.portfolio_school_fraction import features
    from experiments.portfolio_combine import raw_frame
    _, past, future, past_share, errors, weights = training(history, CUTOFF, END)
    mean, components, coordinates, _ = basis(errors, weights)
    # 030's profile predictors use 021 shares; runtime preserves 029 daily totals.
    source = raw_frame(SOURCE / f"raw_{CUTOFF}.csv", CUTOFF, END)
    future = future.copy()
    future["origin"] = pd.Timestamp(CUTOFF)
    future["network_base"] = future.groupby("date").base.transform("sum")
    future["base_share"] = future.base / future.network_base
    future = future.sort_values(["route", "date"]).reset_index(drop=True)
    profile = source.pivot(index=["route", "date"], columns="hour", values="prediction").reindex(columns=HOURS)
    pd.testing.assert_frame_equal(profile.index.to_frame(index=False), future[["route", "date"]])
    total = profile.sum(axis=1).to_numpy()
    shares = np.divide(profile.to_numpy(), total[:, None], out=np.zeros(profile.shape), where=total[:, None] > 0)
    active = future.route.ne(5).to_numpy() & (total > 0)
    if "boardings" in future or not future.date.gt(pd.Timestamp(CUTOFF)).all():
        raise ValueError("Future target entered recipe inputs")
    np.savez_compressed(destination / "inputs.npz", X=np.column_stack([features(past), past_share]),
                        test=np.column_stack([features(future.loc[active]), shares[active]]),
                        coordinates=coordinates, mean=mean, components=components, future_share=shares, active=active)
    save(base, destination / "base.csv")
