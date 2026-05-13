# V1_1 research log: T3 weakening is not removed by new indices or window-length adjustment

## Status

This is a research interpretation log for the V1_1 structural V→P route. It is not a new computation script and does not modify V1 or V1_1 results.

## Question

Does the T3 V→P weakening disappear after either:

1. adding V1_1 structural P/V indices; or
2. changing the T3 window length / boundary?

## Short answer

No. The T3 weakening should be retained as a robust empirical constraint.

Adding structural indices and changing window length both recover some relationships, but neither removes the fact that T3 is a sharp low-density / transition window for V→P relations.

## Evidence from V1_1 structural V→P run

In the full-window V1_1 structural V→P run, T3 remains much weaker than adjacent or major stage windows:

```text
S1 stable_lag_dominant = 56
T1 stable_lag_dominant = 12
S2 stable_lag_dominant = 42
T2 stable_lag_dominant = 47
S3 stable_lag_dominant = 29
T3 stable_lag_dominant = 7
S4 stable_lag_dominant = 19
T4 stable_lag_dominant = 25
S5 stable_lag_dominant = 51
```

The new structural indices do recover a T3 branch, but this recovery is narrow and concentrated:

```text
T3 oldV→oldP stable_lag_dominant = 1–2, depending on baseline table version
T3 newV→oldP stable_lag_dominant ≈ 0–1
T3 oldV→newP stable_lag_dominant = 0
T3 newV→newP stable_lag_dominant = 6
```

The recovered branch is mainly high-latitude structural V→P:

```text
V_pos_north_edge_lat / V_pos_band_width / V_pos_centroid_lat_recomputed / V_highlat_35_55_mean
→ P_highlat_40_60_mean / P_highlat_35_60_mean
```

Therefore, V1_1 supports a local index-projection-mismatch explanation for a high-latitude structural branch, but it does not make T3 broadly recover to the density of S3, S4, or other stronger windows.

## Evidence from V1_1 T3 window-length sensitivity audit

The window-length sensitivity audit tested the current T3 window, equal-length controls, and T3 expansions.

### Current and equal-length controls

```text
T3_current_107_117_len11 stable = 9
S3_center_equal11_092_102 stable = 26
S4_center_equal11_131_141 stable = 4
```

Interpretation:

- T3 being short is relevant, because equal-length controls also change the number of detected pairs.
- But T3 weakness is not explained purely by the fact that it is 11 days long: an 11-day S3 control still has 26 stable pairs, far above T3_current.

### T3 expansion tests

```text
T3_current_107_117_len11 stable = 9
T3_expand_symmetric17_104_120 stable = 11
T3_expand_symmetric23_101_123 stable = 17
T3_expand_backward17_101_117 stable = 19
T3_expand_forward17_107_123 stable = 8
```

Interpretation:

- Expanding T3 can recover additional pairs.
- Recovery is strongest when the window is expanded backward toward S3.
- Forward expansion toward S4 does not recover T3.
- Therefore, the current T3 boundary likely suppresses or cuts off part of the S3→T3 transition signal.

However, even the best T3 expansion does not erase the interpretation that T3 is a transition/weakening window. It partly recovers old-index relationships, but the T3-specific high-latitude structural branch remains narrow and selective.

## Main conclusion

The combined evidence supports the following conclusion:

```text
T3 V→P weakening is not an artifact that disappears after adding structural indices or changing window length.

Adding indices reveals a narrow high-latitude structural V→P branch.
Changing the window, especially expanding backward toward S3, recovers some old-index relationships.
But neither operation removes T3 as a sharp low-density / transition window for V→P relationships.
```

## What this means for interpretation

The T3 drop should not be written as:

```text
V→P physically disappears in T3.
```

It should also not be written as:

```text
The T3 drop is only caused by missing indices.
```

Nor as:

```text
The T3 drop is only caused by the short 11-day window.
```

The more defensible statement is:

```text
T3 marks a robust weakening / reorganization of V→P index-level relationships. Structural indices recover a high-latitude V→P branch, and backward window expansion recovers some old-index relationships, but the abrupt T3 reduction remains. Therefore, T3 should be treated as a real transition/weakening window whose V→P correspondence is partly reorganized and partly window-boundary sensitive, rather than as a simple missing-index or short-window artifact.
```

## Consequences for future work

1. Keep T3 as a scientifically important transition window rather than demoting it because pair counts are low.
2. Do not claim that V1_1 fully repairs T3; it only recovers a specific high-latitude structural branch.
3. When reporting V1/V1_1, explicitly distinguish:
   - robust T3 weakening;
   - recovered high-latitude structural branch;
   - backward-expansion recovery of some old-index pairs;
   - unresolved cause of the remaining weak density.
4. Further work should avoid repeating generic explanations such as “T3 is too short” or “the indices were missing” unless they are attached to the specific evidence above.
