First implementation interrupted before any model metrics were emitted: slow
per-group Python/scipy logsumexp loop. No conclusions from this run. source.py
preserves executed implementation. Replaced by equivalent vectorized grouped
logsumexp for the next run; likelihood, priors and selection unchanged.
