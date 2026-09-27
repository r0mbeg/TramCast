# M20260927-v1, before fitting

Test additional covariates on unchanged B20260927 aggregate-intensity model.
Four routes 1,7,11,12; 315 late-snapshot positions. Fixed Poisson/Gaussian MAP,
prior SD=1, identical features/masks/61-day windows to selected B model.
No new hyperparameter search. Candidate `name`: explicit metro reference in
stop name and interactions with morning/evening/weekend. Candidate `flow` adds
unique nonzero station-line count availability plus centered standardized
log1p quarterly incoming/outgoing counts and the same interactions. Unknown,
ambiguous, zero-review counts are imputed to training-support mean ONLY in the
feature matrix (centered zero), with an availability indicator; not zero demand.
Candidate links and temporal caveats from D20260927 persist. No fake stop truth.

At each cutoff use the last completed quarter from the source. Static snapshot
covariates describe the entire fit and frozen forecast window, NOT a claim of
per-training-day availability. Publication date unknown; retrospective experiment.
Choose base/name/flow by May–June route WAPE; freeze selection for later windows.
Run both new variants on subsequent windows as reported ablations, not new
selection. Base and 030 controls reuse hashed B outputs on identical rows.
Show route WAPE/MAE, coverage, selected share TV vs base, candidate fit gradients.
No stop metrics, no claim of improved linkage or successful BL–FS training.
Final fit through Oct31 only for development-selected candidate. Save profiles
and parameters (not public forecasts), data/code/protocol hashes, results/report.
