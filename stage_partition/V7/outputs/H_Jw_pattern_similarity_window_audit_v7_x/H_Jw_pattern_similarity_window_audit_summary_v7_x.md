# V7-x H/Jw pattern-similarity trajectory window audit

This targeted side audit reads V7-v state-progress curves and checks whether the Jw day30-39 early pattern signal is present in raw R_diff, S_pattern segment advantage, lag alignment, and pattern-only detector scores.

## Key audit questions

### Does Jw show an early day30-39 raw R_diff ramp?
- C0: 0.923473
- C1: 0.958964
- C2: 0.829642
- Baseline sensitivity: positive_across_baselines
- Allowed statement: Jw early pattern-similarity ramp is supported if R_diff gain is positive across C0/C1/C2; this is a pattern-similarity signal, not a profile-object peak.
- Forbidden statement: Do not call it a confirmed Jw object-level transition window without detector/bootstrap support.
- Evidence detail: Jw: C0_full_stage: Rgain=0.9235, Sgain=0.4043, Rshare=0.446; C1_buffered_stage: Rgain=0.959, Sgain=0.3915, Rshare=0.456; C2_immediate_pre: Rgain=0.8296, Sgain=0.409, Rshare=0.431 | H: C0_full_stage: Rgain=0.007149, Sgain=0.4002, Rshare=0.594; C1_buffered_stage: Rgain=0.006747, Sgain=0.3955, Rshare=0.562; C2_immediate_pre: Rgain=0.005665, Sgain=0.3841, Rshare=0.497

### Which field is ahead in S_pattern during early_30_39?
- C0: near_equal
- C1: Jw_ahead
- C2: Jw_ahead
- Baseline sensitivity: baseline_sensitive
- Allowed statement: Use only as segment-level pattern-progress relation.
- Forbidden statement: Do not collapse segment-specific relations into a single whole-window lead.
- Evidence detail: C0_full_stage: meanHminusJw=-0.02986, rel=near_equal; C1_buffered_stage: meanHminusJw=-0.04777, rel=Jw_ahead; C2_immediate_pre: meanHminusJw=-0.03843, rel=Jw_ahead

### Which field is ahead in S_pattern during core_40_45?
- C0: H_ahead
- C1: H_ahead
- C2: mixed
- Baseline sensitivity: baseline_sensitive
- Allowed statement: Use only as segment-level pattern-progress relation.
- Forbidden statement: Do not collapse segment-specific relations into a single whole-window lead.
- Evidence detail: C0_full_stage: meanHminusJw=0.06208, rel=H_ahead; C1_buffered_stage: meanHminusJw=0.06509, rel=H_ahead; C2_immediate_pre: meanHminusJw=0.03451, rel=mixed

### Which field is ahead in S_pattern during late_46_53?
- C0: Jw_ahead
- C1: Jw_ahead
- C2: Jw_ahead
- Baseline sensitivity: stable_across_baselines
- Allowed statement: Use only as segment-level pattern-progress relation.
- Forbidden statement: Do not collapse segment-specific relations into a single whole-window lead.
- Evidence detail: C0_full_stage: meanHminusJw=-0.2082, rel=Jw_ahead; C1_buffered_stage: meanHminusJw=-0.1467, rel=Jw_ahead; C2_immediate_pre: meanHminusJw=-0.1514, rel=Jw_ahead

### Does lag alignment support early Jw pattern leading?
- C0: Jw_pattern_trajectory_earlier_than_H_in_segment
- C1: Jw_pattern_trajectory_earlier_than_H_in_segment
- C2: near_zero_lag
- Baseline sensitivity: baseline_sensitive
- Allowed statement: Early segment lag can indicate local phase offset if best correlations are interpretable.
- Forbidden statement: Do not treat early-segment lag as whole-window phase lead.
- Evidence detail: C0_full_stage: lag=3, corr=0.988, interp=Jw_pattern_trajectory_earlier_than_H_in_segment; C1_buffered_stage: lag=3, corr=0.987, interp=Jw_pattern_trajectory_earlier_than_H_in_segment; C2_immediate_pre: lag=1, corr=0.986, interp=near_zero_lag

### Does pattern-only state detector produce Jw early candidate bands?
- C0: peak_day=20, band=19-21, signal=R_diff
- C1: peak_day=20, band=18-22, signal=R_diff
- C2: peak_day=20, band=19-21, signal=R_diff
- Baseline sensitivity: stable_if_early_candidates_exist_across_C0_C1_C2
- Allowed statement: If present, call these pattern-similarity candidate bands only.
- Forbidden statement: Do not replace V7-w profile-object windows with pattern-only candidates.
- Evidence detail: n_Jw_early_pattern_candidates=8

## Interpretation boundary

V7-x does not replace V7-w profile-object windows. A pattern-similarity candidate or ramp can support a targeted statement about pattern similarity, but it must not be upgraded to a confirmed object-level profile transition window without separate support.