# A20260927-v1, before training

Goal: learn two latent registration-activity regimes, NOT named stops or motion.
Reuse V-v1 anonymous 10s bins, device split, coverage filters and density scores.
Fit on reference streams only, Sep8–11. Use 128 eligible hours selected uniformly
without replacement with seed20260927 (bounded pilot); no held counts in fit.
Select HSMM or fixed20s smoothing on Sep12 held-device log score. Sep13–14 were
already inspected in V and are not a new holdout; omit them from selection.
Freeze parameters/selection before scoring fresh Sep15–21 device streams.
This is contemporaneous reconstruction, not a 61-day forecast. Existing raw
route audits do not provide stop truth or untouched competition evaluation.

HSMM: two alternating states, low/high observed validation activity; Poisson
rates relative to each reference hour's empirical mean. Duration = 1+NB(size2),
capped at120 bins (20min) by collecting the tail in the final bin. Mean durations
and rate contrast learned; rates normalized to unit stationary mean. Initial
state-age distribution is stationary; right-censored final state. Priors on
log contrast/log(mean duration minus1): normal centered at contrast5, durations
120s/30s, SDs1.5/1/1. Bounds: contrast1.1–100, duration means low20–600s,
high12–180s. Priors/bounds are modelling assumptions, not Moscow measurements.
MAP via L-BFGS-B, three fixed starts, max150 iterations, require successful
optimizer and finite objective. Report projected gradient and active bounds.
No extra tuning after validation/holdout. Duration truncation mass reported.

Conditional posterior intensity becomes a normalized within-hour density with
10% uniform mixture, matching V controls. Compare held log score vs fast20s,
slow300s and uniform; cluster bootstrap vehicle-days1000 seed20260927, per-day
stability. Save parameters, soft high-state probabilities and coverage. Phase
counts/durations are inferred registration episodes, not observed stop visits.
No new Moscow stop labels, no stop accuracy, no forecast release.

Fixed independent check: batched HSMM likelihood agrees with previous exact
inference kernel; kernel already tested against complete path enumeration.
Fresh raw preparation must match route12 ledger and prior raw SHA256.
