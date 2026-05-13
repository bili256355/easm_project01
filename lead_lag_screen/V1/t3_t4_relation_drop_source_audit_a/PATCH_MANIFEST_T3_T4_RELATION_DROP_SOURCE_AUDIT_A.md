# Patch manifest: V1 T3/T4 relation drop source audit A

## Added folder

```text
lead_lag_screen/V1/t3_t4_relation_drop_source_audit_a/
```

## Added files

```text
run_v1_t3_t4_relation_drop_source_audit_a.py
README_T3_T4_RELATION_DROP_SOURCE_AUDIT_A.md
PATCH_MANIFEST_T3_T4_RELATION_DROP_SOURCE_AUDIT_A.md
```

## Behavior

- Reads V1 stability-judged result table.
- Writes independent audit outputs under `lead_lag_screen/V1/t3_t4_relation_drop_source_audit_a/outputs`.
- Does not modify V1 code.
- Does not modify existing V1 outputs.
- Does not rerun lead-lag screening.

## Scientific role

This is a result-audit layer for localizing where T3/T4 relationship-count collapse occurs. It is not a mechanism layer.
