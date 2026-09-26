"""Run without torch: python test_chronos_finetune.py."""
import numpy as np
import pandas as pd

from chronos_finetune_experiment import FIT, LORA_FIT, fit_before_cutoff
from pipeline import full_grid


def check():
    history = full_grid().to_frame(index=False)
    history["date"] = pd.to_datetime(history.date)
    history["boardings"] = history.route + history.hour

    class Capture:
        def fit(self, **kwargs):
            return kwargs

    for cutoff, length in [("2025-06-30", 181), ("2025-08-31", 243)]:
        first = fit_before_cutoff(Capture(), history, cutoff, "unused")
        poisoned = history.copy()
        poisoned.loc[poisoned.date.gt(cutoff) | poisoned.route.eq(5), "boardings"] = 99999999
        second = fit_before_cutoff(Capture(), poisoned, cutoff, "unused")
        assert len(first["inputs"]) == 9
        np.testing.assert_array_equal(first["inputs"], second["inputs"])
        assert np.asarray(first["inputs"]).shape == (9, length)
        assert "validation_inputs" not in first
        assert all(first[key] == value for key, value in FIT.items())
        lora = fit_before_cutoff(Capture(), poisoned, cutoff, "unused", lora=True)
        np.testing.assert_array_equal(first["inputs"], lora["inputs"])
        assert all(lora[key] == value for key, value in LORA_FIT.items())
        assert LORA_FIT["finetune_mode"] == "lora" and LORA_FIT["lora_config"]["r"] == 8
    print("Fine-tune checks passed: cutoff isolation, no route 5, daily shape, fixed parameters, no validation in fit.")


if __name__ == "__main__":
    check()
