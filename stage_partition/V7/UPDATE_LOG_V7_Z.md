# UPDATE_LOG_V7_Z

## hotfix_02_summary_writer_hardened_schema

Purpose: fix a runtime crash in `_write_summary()` after hotfix_01 gate hardening.

### Problem

The hotfix_01 classification table switched from legacy metric-row counters to hardened evidence-family fields. The markdown summary writer still indexed legacy columns:

- `n_A_direction_metrics`
- `n_B_direction_metrics`

When the hardened classification table was passed to `_write_summary()`, this raised:

```text
KeyError: 'n_A_direction_metrics'
```

### Fix

`_write_summary()` now supports both schemas:

- legacy classification rows with `n_A_direction_metrics` / `n_B_direction_metrics`;
- hardened classification rows with `hardened_evidence_level` / `hardened_sensitivity_status`.

### Scientific / method impact

None. This is a write-stage hotfix only. It does not change:

- profile construction;
- raw/profile detector;
- shape-pattern detector;
- pre-post state/growth curves;
- bootstrap metrics;
- evidence-family logic;
- final gate decisions.

