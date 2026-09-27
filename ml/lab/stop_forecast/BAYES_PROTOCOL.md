# B20260927-v1 — aggregate intensity, frozen before fitting

User authorizes an isolated aggregate-learning experiment, including Bayesian
regularization, without replacing 030. This extends research scope, not the
verified-stop release gate in PROTOCOL.md. No stop targets are generated.

Support: workbook stop occurrences for routes 1,7,11,12, including direction,
trip_id and sequence. This is a late snapshot, NOT verified 2025 geography.
No exposure, population, employment or actual interchange counts are available.
Features: standardized distance to fixed Moscow centre (55.75,37.62), latitude,
endpoint indicator; spatial interactions with morning (6–10), evening (16–20),
weekend. Shared hour×weekend effects and regularized route intercepts. Weekend
means Saturday/Sunday, not the Russian holiday calendar. Unit exposure omitted,
not interpreted as service frequency. All support coverage is hypothetical.

mu_it=exp(x_it beta), M_rt=sum_i mu_it. Fit unnormalized route likelihood.
Controls: route×hour×weekend training mean and saved 030 forecasts, same mask.
Candidate P: Poisson likelihood + Gaussian prior (MAP equals L2 fit).
Candidate N: negative-binomial aggregate likelihood, fixed size k=10, Gaussian
priors; variance=M+M²/10 is an explicit assumption, not a measured parameter.
Coefficient prior SD 0.25 or 1; route intercept prior SD 2, centred on training
route mean divided by snapshot occurrences (empirical Bayes). Other means zero.
First window chooses likelihood/SD by route WAPE; fixed choices for later windows.
Fits use 3 fixed initializations and retain best posterior objective.

Train through Apr30→May1–Jun30; Jun30→Jul1–Aug30; Aug31→Sep1–Oct31;
final Oct31→Nov1–Dec31. No target update within 61 days. Previously inspected
route windows are NOT independent discovery evidence. Rows with working_events
>0 only; counts have unknown completeness and may be partial. Missing rows are
excluded, not zero-filled. Night structural zeros excluded from metrics; 5 has
only the observed half-hour. No automatic reconstruction of partial counts.

Sensitivity on first window: prior means -1,0,+1 for central-distance coefficient
at chosen SD; compare WAPE and total variation of shares. An exact invariance
check additionally shifts static spatial coefficients and compensates route
intercepts: route sums remain identical for a model without spatial interactions.
This exposes a genuine null direction, not independently validated stop accuracy.
Local rank of derivative of log aggregate mean and Laplace covariance at MAP
are diagnostics. Approximate Gaussian parameter draws (seed20260927, 200) give
conditional intervals only; no claim of calibrated stop uncertainty. Fail on
unconverged optimization or nonpositive posterior curvature.

Outputs: input/code/protocol hashes, parameters, route metrics by window/route/
hour, sensitivity, rank, conditional uncertainty, occurrence feature table,
route predictions and selected final route12 scenario (float expectations).
No publication, submission, production integration or rounding to integers.
Stop metrics remain null; all historical events remain unassigned. Scenarios
sum to their own route forecast; optional saved030 reconciliation evaluated only
as aggregate consistency, never as stop accuracy. No extrapolation to other
routes without historical support and covariates.

Implementation notes: v1 interrupted before metrics due slow grouped evaluation;
v2 used an equivalent vectorized likelihood. v3 subtracts parameter-independent
saturated likelihood constants and refines the optimizer solution, with a maximum
absolute gradient check 0.05. These numerical changes do not change the model,
priors, split or selection criterion; prior run outputs/source are preserved.
Final scenario includes 24 hours (1–4 structural zero), direct expectations,
030-reconciled shares and equal shares of 030 as an unvalidated control. Night
shares/intervals are undefined, not estimates. stop_mode semantics and boarding
permission remain unverified; all 315 snapshot occurrences are hypothetical
support, none is certified as a historically served boarding position.
