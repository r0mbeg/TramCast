# F20260927-v1: independent BL–FS + entropy

Frozen before implementation/run. Scope: independent binary-logit / fractional
multinomial-logit benchmark described in Rahman, Yasmin, Eluru (2020), extended
with KL regularization. NOT a replication of the correlated joint panel model.
Author PDF direct retrieval failed; indexed primary-source excerpts of model
structure and publisher sections were inspected. No Orlando coefficients used.

Training requires complete, provenance-bearing stop counts including observed
zeros, full expected occurrence support, and matching route counts. Missing,
partial, reconstructed or unknown-completeness labels are rejected. Independent
BL loss uses stop indicators y>0. FS uses observed fractions y/sum(y) among
positive stops, each positive route-period equally weighted. Zero-total groups
contribute only to BL. Add lambda*mean_group KL(p||q), q restricted/renormalized
to the same positive support, plus fixed L2=0.01 on coefficients for stability.
BL receives an intercept; FS does not (it cancels within a route-period).
Inference: BL>=0.5 defines predicted positive support, then FS normalizes over
it. If every stop is excluded, the route total stays unassigned. This hard gate
is an explicit operational adaptation, not a calibrated expectation over gates.
No panel random effects, alighting or passenger occupancy model.

No usable Moscow stop labels are available. Do not fit the supervised model on
route-derived pseudo-labels. Test fixtures verify implementation only and never
enter research metrics, model artifacts or a reported accuracy estimate.

Separate PRIOR-ONLY scenario diagnostic reuses frozen B20260927-v4 log utilities
as an unvalidated base s, with ALL snapshot positions assumed available and BL
probability unavailable. It solves KL(p||s)+lambda KL(p||q), analytically:
p proportional to exp((log(s)+lambda log(q))/(1+lambda)). This is a separate
inference projection, not training BL–FS and not the supervised objective above.
Compare lambda=0,0.1,1,10,100 and q=uniform / mild centre-favouring scenario
q proportional exp(-0.5*standardized_distance). No prior is asserted true.
Measure entropy, KL to q, TV to base and prior sensitivity, not stop accuracy.
Do not select lambda using route WAPE: normalized shares have identical totals.

Use unchanged temporal windows and 030 forecasts from B20260927: 61 days per
window, no within-horizon target update. Export one clearly marked illustrative
route12 scenario for Nov–Dec at lambda=1, q=uniform, not an accuracy-selected
model. Include structural night zeros, null activity probabilities, provenance,
float expectations, model/version and input hashes. Preserve old outputs/030.
