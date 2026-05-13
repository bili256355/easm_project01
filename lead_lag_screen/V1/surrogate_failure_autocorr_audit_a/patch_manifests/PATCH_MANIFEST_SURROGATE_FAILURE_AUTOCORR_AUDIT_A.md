# Patch manifest: surrogate_failure_autocorr_audit_a

## Purpose

Add an independent V1 audit to diagnose why many positive-lag candidates are rejected by surrogate/FDR support, especially in T3/T4.

## Added files

- `lead_lag_screen/V1/surrogate_failure_autocorr_audit_a/run_v1_surrogate_failure_autocorr_audit_a.py`
- `lead_lag_screen/V1/surrogate_failure_autocorr_audit_a/README_SURROGATE_FAILURE_AUTOCORR_AUDIT_A.md`
- `lead_lag_screen/V1/surrogate_failure_autocorr_audit_a/patch_manifests/PATCH_MANIFEST_SURROGATE_FAILURE_AUTOCORR_AUDIT_A.md`

## Non-goals

- Does not modify V1 main screening code.
- Does not overwrite V1 outputs.
- Does not replace V1 surrogate p-values.
