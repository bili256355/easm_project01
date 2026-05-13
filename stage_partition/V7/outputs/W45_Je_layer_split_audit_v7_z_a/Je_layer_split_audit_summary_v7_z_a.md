# Je Layer-Split Audit for W45

Version: `v7_z_je_audit_a`
Generated: `2026-05-05T14:43:04`

## 1. Purpose
Audit whether Je's early shape-pattern peak around day33 and late raw/profile peak around day46 represent a credible layer split or a normalization/detector artifact.

## 2. Main decision
- Final Je type: `shape_normalization_sensitive`
- Shape peak reliability: `shape_peak_normalization_sensitive`
- Raw/profile peak reliability: `raw_peak_reliable`
- Raw-vs-shape peak separation: `raw_later_than_shape_supported`

## 3. Peak references from V7-z
- raw_profile: CP001 peak day 46, window 36-56, support 0.978, class accepted_window
- raw_profile: CP002 peak day 26, window 16-36, support 0.825, class candidate_window
- raw_profile: CP003 peak day 33, window 23-43, support 0.763, class weak_window
- shape_pattern: CP001 peak day 33, window 23-43, support 0.957, class accepted_window
- shape_pattern: CP002 peak day 20, window 14-30, support 0.794, class weak_window
- shape_pattern: CP003 peak day 45, window 35-55, support 0.914, class candidate_window

## 4. Normalization audit
- Day30–36 low-norm warning days: [30, 31, 32, 33, 34]
- Minimum day30–36 norm quantile rank: 0.015

## 5. Before/after profile metrics
- day33_shape_peak: dominant=amplitude_or_raw_state_dominated, raw_l2=3.523, shape_l2=0.3326, raw_amp_change=2.363, raw_axis_change=2, shape_axis_change=2
- day46_raw_peak: dominant=amplitude_or_raw_state_dominated, raw_l2=2.185, shape_l2=0.1362, raw_amp_change=0.7663, raw_axis_change=0, shape_axis_change=0

## 6. Bootstrap separation
- status=available; delta_median=13.0; delta_q025=10.0; delta_q975=17.0; P_raw_later_than_shape=1.0; decision=raw_later_than_shape_supported

## 7. Allowed statement
Je's early shape-pattern signal should be treated as normalization-sensitive and should not be upgraded to a stable pattern-preconditioning result.

## 8. Forbidden statement
- Do not write that Je overall leads all objects.
- Do not write that Je drives P/V/H/Jw.
- Do not collapse shape-pattern timing and raw/profile timing into one transition day.
