# V9.1_e hotfix01: safe empty-table concatenation

## Scope
- Patch target: `targeted_svd_order_mode_audit_v9_1_e.py`.
- Fixes the cross-window aggregation crash when a collection such as `all_skipped` contains only empty DataFrames.

## Bug
The helper `cat(parts)` checked only whether the original list was non-empty, then passed an empty filtered list to `pd.concat`, causing:

```text
ValueError: No objects to concatenate
```

## Fix
`cat(parts)` now filters non-empty DataFrames first and returns an empty DataFrame if none remain.

## Method status
- No change to targeted SVD / MCA logic.
- No change to permutation, LOO, CV, group peak, or evidence rules.
- No change to output naming, except previously crashing empty aggregate outputs now write empty CSVs normally.
