# ROOT_LOG_03 append — V10.7_j

## V10.7_j status

V10.7_j adds a new single-entry audit:

```text
stage_partition/V10/v10.7/scripts/run_w45_snr_reliability_v10_7_j.py
```

Output directory:

```text
stage_partition/V10/v10.7/outputs/w45_snr_reliability_v10_7_j
```

## Purpose

V10.7_j is a reliability / signal-to-noise audit for the yearwise scalar object-window indicators used by V10.7_i. It does not test a new physical mechanism. It checks whether V10.7_i's E2→M non-detection should be interpreted as a meaningful scalar-mapping negative result or downgraded to low-power / non-decisive due to unstable object-window indicators.

## Execution chain

```text
smoothed_fields.npz
→ P/V/H/Je/Jw object daily strengths for E1/E2/M
→ object-window SNR and bootstrap ranking reliability
→ leave-one-day-out pairwise mapping sensitivity
→ ±2-day E2/M window-shift sensitivity
→ route decision table
```

Je/Jw are derived from u200 sectors:

```text
Je = u200, 120–150E, 25–45N, jet_q90_strength
Jw = u200, 80–110E, 25–45N, jet_q90_strength
```

## Boundary

This version must not be cited as evidence for or against H's physical role in W45. It only evaluates whether the current scalar yearwise mapping setup has enough reliability to support interpretation.
