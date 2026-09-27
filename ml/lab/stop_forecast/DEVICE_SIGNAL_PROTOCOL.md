# V20260927-v1: fixed before measuring signal

Use Q discovery week Sep8–14 and route12. This is contemporaneous
reconstruction research, NOT a 61-day forecast and NOT stop truth.
Exclude invalid device/vehicle keys, conflicting device→vehicle calendar days,
and any vehicle calendar day carrying another route in the full week data.
Do not deduplicate successful payments or use card identifiers.

For each remaining vehicle-day select lexicographically first device as held
device (fixed by ID, not by fit); others form reference. Split into full working
calendar hours 0,6–23; hour5 excluded to avoid a partial-hour support.
Keep groups with >=10 held and >=20 reference events in that hour; report
all exclusions. Counts conditional on observed event totals; no claim of
counter completeness or known device uptime. No reset at every payment gap.

10-second bins, 360 bins per hour. Reference-count density smoothed with
Gaussian sigma20s (candidate), sigma300s (slow control), and uniform control.
Reflection at hour boundaries; normalized density with 10% uniform mixture.
No fitted bandwidth or clock shift. Score held events by mean log density
ratio in nats/event. Placebo uses circular shifts of the reference histogram
by 10,20,30,40,50 minutes, then applies the same fast smoothing.
These controls preserve within-hour count/burst content, not geography.

Report weighted log-score differences, per-day stability, and vehicle-day
cluster bootstrap 95% interval (1000 resamples, seed20260927). Calendar-day
correlation across vehicles remains, so bootstrap intervals are descriptive,
not a causal/generalization guarantee. Held-device simultaneity alone may
reflect traffic, boarding waves, clock artifacts, or shared reporting outages.
It cannot establish stop identity, travel time or complete registrations.

Separately implement finite HSMM inference using supplied Poisson rates and
duration distributions. Verify exact marginalization against enumeration,
unknown-phase symmetry, supplied boundary anchor, missing observations.
No fitted Moscow HSMM, synthetic stop-score, or forecast release in this stage.
