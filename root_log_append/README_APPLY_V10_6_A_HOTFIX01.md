# V10.6_a HOTFIX01: NumPy trapezoid compatibility

## Purpose

Fix a runtime crash in `curve_metrics.py` caused by environments where `numpy.trapz` is no longer available.

Observed error:

```text
AttributeError: module 'numpy' has no attribute 'trapz'. Did you mean: 'trace'?
```

## Files to replace

Copy this file into the existing V10.6_a package, replacing the old file:

```text
stage_partition/V10/v10.6/src/stage_partition_v10_6/curve_metrics.py
```

## What changed

The AUC integration call was changed from direct `np.trapz(...)` to a small compatibility wrapper:

1. use `np.trapezoid(...)` when available;
2. fall back to `np.trapz(...)` on older NumPy versions;
3. fall back to a manual trapezoidal calculation if neither API exists.

This does not change the intended AUC semantics.

## Run command after replacement

```bash
python D:\easm_project01\stage_partition\V10\v10.6\scripts\run_w045_precluster_audit_v10_6_a.py
```

## Status

Syntax/import-chain check was run on the replacement package files.
This hotfix only addresses the NumPy API compatibility bug. It does not change the experiment design, entry script, output directory, or interpretation rules.
