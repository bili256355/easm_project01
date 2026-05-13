# Patch manifest: smooth5 strict index path hotfix

## Scope
Replaces only:

`lead_lag_screen/V1/surrogate_failure_autocorr_audit_a/run_v1_surrogate_failure_autocorr_audit_a.py`

## Reason
The previous auto-discovery could silently select an index file from a different smoothing basis when both smooth5 and smooth9 outputs exist. That would invalidate the AR1/null-threshold audit.

## Change
- Disables broad recursive index auto-discovery for default runs.
- Defaults only to the exact V1 smooth5 anomaly table:
  `D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a\indices\index_anomalies.csv`
- If that exact file is unavailable or fails validation, the script now fails loudly and asks for `--index-csv`.

## Validation
`python -S -m py_compile` passed for the patched script.
