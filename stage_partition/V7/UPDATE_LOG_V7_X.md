# UPDATE_LOG_V7_X

## V7-x: H/Jw pattern-similarity trajectory window audit

Purpose:
- Add a narrow targeted side audit for the early Jw pattern-similarity signal found in V7-v curves.
- This version reads existing V7-v `w45_H_Jw_state_progress_curves_v7_v.csv`; it does not recompute raw fields.
- It checks whether the Jw day30–39 early signal is present in raw `R_diff`, normalized `S_pattern`, segment-level H/Jw advantage, lag alignment, and pattern-only trajectory detector scores.

Scope:
- Fields: H and Jw only.
- Baselines: C0_full_stage, C1_buffered_stage, C2_immediate_pre.
- Signals: R_diff and S_pattern.
- Day range: day0–70.

Important interpretation boundary:
- V7-x is not a replacement for V7-w profile-object window detection.
- If V7-x finds an early pattern-similarity ramp or candidate band, it supports only a pattern-similarity statement.
- Do not upgrade it to a confirmed profile-object transition window without separate profile-object support.

New files:
- scripts/run_H_Jw_pattern_similarity_window_audit_v7_x.py
- src/stage_partition_v7/H_Jw_pattern_similarity_window_audit.py

Expected outputs:
- outputs/H_Jw_pattern_similarity_window_audit_v7_x/H_Jw_pattern_similarity_trajectory_curves_v7_x.csv
- outputs/H_Jw_pattern_similarity_window_audit_v7_x/H_Jw_early_pattern_ramp_audit_v7_x.csv
- outputs/H_Jw_pattern_similarity_window_audit_v7_x/H_Jw_pattern_segment_advantage_v7_x.csv
- outputs/H_Jw_pattern_similarity_window_audit_v7_x/H_Jw_pattern_lag_alignment_v7_x.csv
- outputs/H_Jw_pattern_similarity_window_audit_v7_x/H_Jw_pattern_state_detector_peaks_v7_x.csv
- outputs/H_Jw_pattern_similarity_window_audit_v7_x/H_Jw_pattern_state_detector_scores_v7_x.csv
- outputs/H_Jw_pattern_similarity_window_audit_v7_x/H_Jw_pattern_growth_windows_v7_x.csv
- outputs/H_Jw_pattern_similarity_window_audit_v7_x/H_Jw_pattern_similarity_window_audit_summary_v7_x.csv
- outputs/H_Jw_pattern_similarity_window_audit_v7_x/H_Jw_pattern_similarity_window_audit_summary_v7_x.md
- outputs/H_Jw_pattern_similarity_window_audit_v7_x/run_meta.json
