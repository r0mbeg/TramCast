# Dense distillation: predeclared protocol

User request: improve CPU-only 61-day forecasting with GPU-generated targets for arbitrary 2025 starts.
GPU teacher: frozen recipe 030, weekly cutoff dates 2025-04-09 to 2025-10-29;
additional 2025-05-31 reproduction check. One GPU, two CPU cores, 3600-second batch budget
(excluding the 41-second pilot). Existing teacher inputs and parameters stay unchanged.
Each new teacher receives history physically truncated at its origin. External sources
are the existing retrospective weather, production/school calendars and movement snapshots.
All old recipe selection is retrospective; these are research comparisons, not independent tests.

CPU candidates fixed before new scores: CatBoost depth 7 / 800 trees / RMSE and
LightGBM 31 leaves / 800 trees / L1, learning rate .05, two CPU threads, seed 42.
Teacher-weight fractions: 1.0, 0.75, 0.5; plus enhanced observed-only LightGBM control.
The student predicts counts divided by an observed historical calendar-hour profile.
Weights deduplicate overlapping route/date/hour targets; teacher and actual groups
are separately normalized before mixing. No hidden target observations are generated.
Features: existing CPU history/calendar/movement plus calendar-adjusted profiles,
annual/hour cycles, weather and school calendar. No teacher prediction is an inference feature.

Development starts: 2025-07-01 and 2025-08-01, complete 61-day horizons.
Choose lowest mean observed WAPE after structural zeros and half-up rounding.
Each fit uses actual observations strictly before the start and teacher origins
strictly before the start. Teacher forecasts may cover later dates: they are
pseudo-labels available from an earlier origin, NOT actual future counts.
Test start: 2025-09-01, full September-October; no re-selection from test results.
Compare teacher fidelity and observed WAPE separately, same observed mask.

Additional stricter control: selected student trained with all teacher target dates
also ending before September 1 (purged future pseudo-targets). This measures transfer
without teacher forecasts covering the evaluation period. Do not conflate its result
with teacher-assisted forecasting in the primary protocol.

Final refit: actual data through October 31, teachers with origins through October 29.
Do NOT include the existing November 1 teacher in fit. Output November-December submission.
January-March and 2026 are not quality-validated. Final weights are retrospective
for earlier 2025 starts. Unknown future external snapshots remain missing, not fabricated.

Checks: exact 14640-row grid, int64/nonnegative/half-up, route5 and hours1-4 zero,
forecast unchanged by poison of future actual values/masks, native model reload,
model/source hashes, CPU-only dependencies, fresh-process inference resources.
CPU budget: 1800 seconds. No runtime/Go/UI changes, no commits.

User-reported external controls: observed CatBoost 0.85630; LightGBM Laplace 0.84716.
These closed scores are not optimization inputs and cannot be locally reproduced.

Pre-training implementation amendment: the legacy regime parent fails at June 4
because summer history does not yet contain every weekday. New teacher generation
uses a per-route/weekday/hour fallback to pre-origin all-season history only for
missing seasonal groups. No replacement observations or future facts are inserted.
This is a 030-family extension for new dates; existing recipe/runtime remains frozen.
May 31 reproduction before this extension matched all 14640 published values exactly.
Generation resumes with a new source manifest; old completed targets remain immutable.

Exact target inventory: 30 weekly cutoffs April 9–October 29 plus May 31 = 31
teachers / 453840 full-grid rows. Trainer refuses incomplete inventory.
RMSE ratio-model weights use squared reference counts to represent count-space
squared loss; L1 uses reference counts. Both correct repeated physical targets
and normalize teacher/actual weight groups before applying alpha.

Teacher fidelity is additionally measured against the existing exact September 1
and November 1 030 forecasts. Neither exact-origin forecast enters student training
or candidate selection. November fidelity is not accuracy against hidden facts.

Execution is split into development and finish so CPU selection can run while later
GPU origins are generated. Development requires all teacher origins through July 31;
finish requires all 31. The frozen selection and development source hashes are checked
before finish. Total CPU wall budget remains 1800 seconds across the two phases.
