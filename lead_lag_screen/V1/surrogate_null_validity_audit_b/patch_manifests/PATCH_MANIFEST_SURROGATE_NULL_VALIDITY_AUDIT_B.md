# Patch manifest: surrogate_null_validity_audit_b

## Added files

- `lead_lag_screen/V1/surrogate_null_validity_audit_b/run_v1_surrogate_null_validity_audit_b.py`
- `lead_lag_screen/V1/surrogate_null_validity_audit_b/README_SURROGATE_NULL_VALIDITY_AUDIT_B.md`
- `lead_lag_screen/V1/surrogate_null_validity_audit_b/patch_manifests/PATCH_MANIFEST_SURROGATE_NULL_VALIDITY_AUDIT_B.md`

## Scope

Standalone V1 audit. Reads V1 stability judgement output and smooth5 index anomalies. Writes to its own output folder.

## Safety

- Does not modify V1 main-screen code.
- Does not modify existing V1 outputs.
- Does not broad-search index CSVs, to avoid smooth5/smooth9 mix-ups.

## Syntax check

`python -S -m py_compile` passed for the new entry script.
