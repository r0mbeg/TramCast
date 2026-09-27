# P20260927-v1 — fixed before fitting

One candidate, unchanged B Poisson/Gaussian MAP (SD1), four routes 1/7/11/12.
Append log1p(nearest entrance great-circle distance in metres) and
log1p(distinct station-line candidates within 500m), standardized across the
fixed 315-position catalog, with morning6–10/evening16–20/weekend interactions.
No radius/penalty/feature search. Zero candidates describes the 2021 catalog,
not confirmed absence of a metro station in 2025. Use hashed G-v1 covariates;
retain its historical and walking-access limitations. No quarterly flows.

Reuse metro_fit (three starts, gradient guard .05) and frozen B windows:
train through Apr30 / Jun30 / Aug31, predict the following 61 days without
new target facts. Reuse identical B observation masks and base/030 controls.
Select base/proximity by May–June WAPE, freeze selection; later windows are
stability ablations, not a second selection. They were explored in prior work
and are not an untouched independent holdout. Fit Oct31 only if proximity wins
the first window. Failed convergence has null metrics, never silently a forecast.

Report route WAPE/MAE, per-route and per-hour errors, observation coverage,
share TV vs base and aggregate consistency. Save profiles for every successful
fit as unvalidated scenarios, not stop observations. No stop MAE/WAPE or
uncertainty calibration without truth. Do not modify prior runs or runtime.
