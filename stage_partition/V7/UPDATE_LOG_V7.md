# V7 Update Log

## V7-e2 — progress_order_failure_audit_v7_e2

Added a lightweight failure-decomposition audit for V7-e / V7-e1 progress-order results.

Scope:
- Reads existing V7-e progress timing outputs and V7-e1 pairwise delta significance outputs.
- Does **not** recompute progress timing.
- Does **not** change V7-e implementation, bootstrap samples, LOYO samples, or pairwise significance rules.
- Does **not** introduce any minimum effective day threshold.

Main purpose:
- Explain why most pairwise progress orders do not pass the 90% bootstrap CI criterion.
- Separate failures into central overlap, tail uncertainty, progress-quality bottleneck, LOYO conflict, or method-resolution-risk cases.

New entry script:
- `stage_partition/V7/scripts/run_progress_order_failure_audit_v7_e2.py`

New module:
- `stage_partition/V7/src/stage_partition_v7/progress_order_failure_audit.py`

Main outputs:
- `pairwise_progress_failure_audit_v7_e2.csv`
- `window_progress_failure_audit_v7_e2.csv`
- `pairwise_progress_tail_samples_v7_e2.csv`
- `progress_order_failure_audit_v7_e2.md`
- `run_meta.json`

Interpretation boundary:
- This is a failure audit, not a new result-picking layer.
- Pairs that fail 90% should not be upgraded by narrative wording.
- `method_resolution_limit_risk` is a caution flag indicating where region/component-level progress may be worth checking, not a confirmed diagnosis.
