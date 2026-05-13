# T3 P/V Latitudinal Object Change Audit v1_a - trapezoid hotfix

## Purpose

Fix a NumPy compatibility error in `t3_p_v_latitudinal_object_change_core.py` where `np.trapz` is unavailable in the user's NumPy build.

## Error fixed

```text
AttributeError: module 'numpy' has no attribute 'trapz'. Did you mean: 'trace'?
```

## Changed file

```text
src/lead_lag_screen_v1/t3_p_v_latitudinal_object_change_core.py
```

## Change details

- Added `safe_trapezoid(y, x)` wrapper.
- Uses `np.trapezoid` when available.
- Falls back to legacy `np.trapz` when available.
- Falls back to a local manual trapezoidal integration implementation if neither NumPy API is exposed.
- Replaced all direct `np.trapz(...)` calls in the file.

## Scientific semantics

This hotfix does not change the intended calculation semantics. It only restores trapezoidal latitude-profile integration under the current NumPy version.
