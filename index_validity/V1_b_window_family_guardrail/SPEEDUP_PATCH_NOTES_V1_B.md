# index_validity/V1_b window-family guardrail speedup patch

This is an equivalent-performance patch for `index_validity/V1_b_window_family_guardrail`.
It does not change the scientific scope or default data semantics:

- default data mode remains `smoothed`;
- default output tag remains `window_family_guardrail_v1_b_smoothed_a`;
- quantiles, bootstrap replicate count, EOF mode count, scoring thresholds, family guardrail rules, and output table names are unchanged;
- the patch only reorganizes computation for speed and adds runtime timing outputs.

## Speedups implemented

1. Cache `(window, family)` field subsets.
2. Cache repeated sample extraction by `(family, year/day signature)`.
3. Cache EOF/SVD results for repeated `(window, family, sample set, n_modes)` use.
4. Replace per-grid Python-loop field R² with vectorized pairwise-correlation algebra.
5. Replace bootstrap composite re-materialization with year-level sum/count aggregation.
6. Add `tables/runtime_task_timing.csv` for diagnosing future bottlenecks.

## Optional runtime controls

Defaults preserve previous output behavior. Optional flags can be used for diagnostic speed runs:

```bat
python index_validity\V1_b_window_family_guardrail\scripts\run_index_validity_window_family_guardrail_v1_b.py --tables-only
python index_validity\V1_b_window_family_guardrail\scripts\run_index_validity_window_family_guardrail_v1_b.py --no-cartopy
python index_validity\V1_b_window_family_guardrail\scripts\run_index_validity_window_family_guardrail_v1_b.py --max-figures 40
```

These flags affect plotting only, not table metrics.
