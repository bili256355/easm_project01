# Patch manifest: transition_window_sensitivity_audit_a

## Added

```text
lead_lag_screen/V1/transition_window_sensitivity_audit_a/run_v1_transition_window_sensitivity_audit_a.py
lead_lag_screen/V1/transition_window_sensitivity_audit_a/README_TRANSITION_WINDOW_SENSITIVITY_AUDIT_A.md
lead_lag_screen/V1/patch_manifests/PATCH_MANIFEST_TRANSITION_WINDOW_SENSITIVITY_AUDIT_A.md
```

## Scope

Adds an independent V1 audit layer. It only reads existing V1 stability results and smooth5 index anomalies, then writes a new output directory under the audit folder.

It does not modify:

```text
lead_lag_screen/V1 main scripts
lead_lag_screen/V1 existing outputs
foundation/V1 outputs
```

## Main command

```bat
python D:\easm_project01\lead_lag_screen\V1\transition_window_sensitivity_audit_a\run_v1_transition_window_sensitivity_audit_a.py
```

## Checks performed before packaging

```text
python -S -m py_compile run_v1_transition_window_sensitivity_audit_a.py
```
