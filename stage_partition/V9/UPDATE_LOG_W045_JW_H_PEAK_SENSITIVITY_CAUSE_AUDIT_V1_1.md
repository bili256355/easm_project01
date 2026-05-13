# UPDATE_LOG: W045 Jw/H Peak Sensitivity Cause Audit v1_1

## Purpose

This patch adds a narrow correction audit on top of `peak_sensitivity_cause_audit_w045_jw_h_v1`.
It does not redefine W045, does not rerun V9 peak detection, and does not perform full physical subpeak classification.

## Why v1_1 is needed

The v1 result suggested that:

- H sensitivity is likely affected by max_score and outside-system/search-window candidates.
- Jw has near-window statistical clusters, but physical distinctness was not closed.
- Jw-H all-config order looked stable, but this may be inflated by inadmissible, outside-system, or rule-locked configurations.

v1_1 therefore adds:

1. config-level admissibility screening;
2. filtered Jw-H order recomputation;
3. Jw max-lat/core-lat proxy axis shift audit;
4. H candidate legality audit;
5. cluster distinctness revision;
6. revised v1_1 diagnosis table.

## New entry

```bat
python D:\easm_project01\stage_partition\V9\scripts\run_peak_sensitivity_cause_audit_w045_jw_h_v1_1.py
```

## New output directory

```text
V9/outputs/peak_sensitivity_cause_audit_w045_jw_h_v1_1
```

## Interpretation constraints

- `core_lat` in this patch is a proxy based on v1 `max_lat`, because v1 did not save full profile vectors.
- A Jw `AXIS_SHIFT_CLEAR` flag is only follow-up evidence, not a confirmed physical subpeak.
- H outside-system or max-score locked clusters should not enter W045 internal subpeak/order interpretation.
- All-config Jw-H order should not be used as a result if filtered/core layers disagree.
