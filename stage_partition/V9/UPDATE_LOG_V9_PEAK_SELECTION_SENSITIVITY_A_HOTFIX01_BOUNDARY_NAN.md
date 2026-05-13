# V9_peak_selection_sensitivity_a hotfix01: boundary NaN warning handling

## Purpose
Fix repeated `RuntimeWarning: Mean of empty slice` warnings caused by leading/trailing all-NaN days produced by running-window smoothing.

## Change
- Added `_nanmean_no_warning`, a NaN-aware mean helper that keeps all-NaN slices as `NaN` without emitting warnings.
- Replaced direct `np.nanmean` calls in:
  - fallback longitude/profile averaging,
  - year-to-composite profile averaging,
  - detector pre/post contrast averaging.

## Scientific semantics
- No valid data are imputed.
- Boundary all-NaN days remain invalid and continue to be excluded by detector validity logic.
- This is an engineering/noise-control hotfix only; it does not change the intended peak-selection sensitivity design.

## Run command
```bat
python D:\easm_project01\stage_partition\V9\scripts\run_peak_selection_sensitivity_v9_a.py
```
