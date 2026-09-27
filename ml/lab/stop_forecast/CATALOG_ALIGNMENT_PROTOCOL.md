# C20260927-v1: catalog and conditional episode alignment

Audit data/catalog/catalog.xlsx against the three original geography sheets;
inspect OSM stop-role support and timestamps. Read only, no backend changes.
Build route12 occurrence graph from catalog, never collapse repeated stops.

Use frozen A-v1 holdout high-state probabilities, Sep15–21. Contiguous bins
with P(high)>=.5 form registration episodes. Hour-boundary episodes are censored.
Episode time = reference-count weighted mean of bin centres; zero reference
weight falls back to interval midpoint. Held counts do not affect alignment.
Counts in episodes remain time-aggregated registrations, NOT stop labels.

Conditional scenario: concatenate direction0 then direction1 into a cycle.
Two terminal transitions are explicit assumptions, not catalog trip links.
Edge length = great-circle distance (lower geometric proxy, not rail geometry).
Forward steps0–8 have a fixed truncated geometric prior with p=.35. Step0
mean elapsed20s. Other elapsed means: summed edge proxy / speed + dwell*steps
+60s per assumed terminal transition, minimum1s. LogNormal sigma=.5, specified
mean converted to log scale. Scenarios speeds10/15/20kmh × dwell0/15s. No tuning.
Initial phase uniform across97 occurrences (hence direction prior50/97 vs47/97),
all phases retained via exact forward/backward. Missing visits allowed by skips;
clock/payment delays only approximated by broad elapsed likelihood here.

Choose64 eligible hours (>=6 uncensored episodes) uniformly seed20260927.
Require >=64, otherwise use all and report. Reconstruct same episodes under
all6 scenarios, compare phase certainty and TV sensitivity; scores conditional
on chosen geography/timing assumptions, not calibrated stop probabilities.
No selecting a scenario by likelihood on this already explored test week.

Hard publication/observation gate: late network validity unverified, historical
timing anchors absent and no independent stop truth -> accepted assignments0.
Save candidates in separate scenario tables, never update unassigned ledger.
Exact small-chain enumeration test; test missing phases remain ambiguous.
Synthetic anchored example tests algorithm only, no Moscow accuracy claim.
