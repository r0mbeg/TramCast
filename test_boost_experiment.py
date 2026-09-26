"""Run: python test_boost_experiment.py."""
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from boost_experiment import FEATURE_SETS, PARAMETERS, estimator, forecast, matrix
from model import training_examples
from pipeline import KEYS, full_grid


def check():
    data = full_grid("2025-01-01", "2025-05-31").to_frame(index=False)
    data["date"] = pd.to_datetime(data.date)
    data["boardings"] = data.route * 10 + data.hour + data.date.dt.dayofweek
    cutoff = "2025-04-30"
    keys = data.loc[data.date.gt(cutoff), KEYS]
    x, y, _ = training_examples(data, cutoff)
    poisoned = data.copy()
    poisoned.loc[poisoned.date.gt(cutoff), "boardings"] = 999999
    for algorithm in PARAMETERS:
        for selected in FEATURE_SETS.values():
            model = estimator(algorithm, rounds=2).fit(matrix(x, selected, algorithm), y)
            original = forecast(model, data, keys, cutoff, selected, algorithm)
            pd.testing.assert_frame_equal(original, forecast(model, poisoned, keys, cutoff, selected, algorithm))
            pd.testing.assert_frame_equal(original, forecast(pickle.loads(pickle.dumps(model)), data, keys, cutoff, selected, algorithm))
            pd.testing.assert_frame_equal(original[KEYS], keys.reset_index(drop=True))
            assert original.prediction.dtype.kind == "i" and np.isfinite(original.prediction).all()
            assert original.prediction.ge(0).all()
            assert original.loc[original.route.eq(5) | original.hour.between(1, 4), "prediction"].eq(0).all()
            if algorithm != "catboost":
                a = matrix(x, selected, algorithm)
                b = matrix(x[x.route.eq(50)], selected, algorithm)
                assert a.loc[x.route.eq(50), "route"].cat.codes.eq(b.route.cat.codes).all()
    print("Boost checks passed: six model paths, category mapping, cutoff, persistence, keys, integer zeros.")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        check()
