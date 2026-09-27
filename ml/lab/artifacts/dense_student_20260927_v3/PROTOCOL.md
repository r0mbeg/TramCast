# Boundary-aware distillation, diagnostic follow-up

This is a follow-up AFTER viewing v2 September–October results, not an independent
test. No new hyperparameter selection: keep CatBoost depth7, 800 trees, RMSE,
learning_rate .05, seed42, two CPU threads and equal teacher/actual loss weights.

Diagnosis: v2 copies its August27 teacher on September1–October27 with 2.0132%
WAPE; that teacher's observed score is .87843 versus .89908 for exact August31
teacher on the same dates. June30's 61-day calibration window completes August30,
so a weekly August27 snapshot misses an important change in the teacher context.
October31 similarly completes the August31 window. This is a discrete teacher
state change, not merely a weak student or a few days of ordinary drift.

Add two GPU origins August30 and September30 (one GPU, <=300 seconds), and reuse
existing pinned 030 origins June30, July31, August31 and October31. Existing May31
is already present. New inventory: 37 full teachers. Add the number of completed
monthly 61-day windows as a CPU-only context feature; it depends on dates only.

Diagnostic evaluation: fit actual data through August31, teacher origins strictly
before August31; the exact September1 teacher remains excluded. In particular,
August30's prediction uses no September observations. Re-evaluate September–October
once, no tuning against it. Compare fidelity, actual WAPE and preceding v2.

Final fixed-year compression: fit actual data through October31 and all 37 teachers,
INCLUDING October31's original 030 forecast for November–December. This is authorized
teacher-generated supervision, not hidden actual counts. November teacher fidelity
is therefore TRAINING fidelity and must not be described as held-out generalization.
Held-out arbitrary-origin fidelity: use existing August14 030 teacher, excluded from
all 37 origins. Final all-year weights make this retrospective interpolation, not a
causal as-of forecast. No observed August accuracy claim from that final model.

Native CPU bundle and distinct submission; v2 outputs are retained. Early-year,
2026, changed-history and retrospective weather limitations persist. New budget:
600 CPU seconds for two fixed fits and checks, no parameter search. Source hashes,
future-fact poison, full-grid/zeros, reload and fresh-process CPU inference checked.
No SOTA claim without independent closed-test evidence.
