# ROOT_LOG_05 append — V10.7_j

## Pending task resolved by V10.7_j

V10.7_i produced a non-detection for stable E2→M scalar transition mapping. The pending issue is whether that non-detection is meaningful or simply reflects poor signal-to-noise, fixed-window timing error, or unreliable yearwise object-window rankings.

V10.7_j addresses this by auditing:

- object-window SNR,
- bootstrap ranking reliability,
- leave-one-day-out sensitivity,
- ±2-day window-shift sensitivity.

## Forbidden interpretations

Do not write:

```text
V10.7_j proves E2 and M are connected / disconnected.
V10.7_j proves H has / lacks W45 influence.
V10.7_j replaces structural or position/shape mapping tests.
```

Allowed interpretation:

```text
V10.7_j evaluates whether the scalar indicators used in V10.7_i are reliable enough for yearwise E2→M mapping. If reliability is low, V10.7_i's non-detection must be treated as non-decisive.
```

## Recommended next decision

If V10.7_j finds low SNR or high window sensitivity, stop interpreting V10.7_i as a negative mapping result and move to either peak-aligned windows or position/shape transition vectors. If V10.7_j finds high reliability, then V10.7_i's scalar non-detection becomes more interpretable, while still not excluding non-scalar H roles.
