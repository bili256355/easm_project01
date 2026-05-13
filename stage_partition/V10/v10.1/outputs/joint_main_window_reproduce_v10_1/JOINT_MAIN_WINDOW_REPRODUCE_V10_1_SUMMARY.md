# V10.1 joint main-window reproduction summary

This run semantically rewrites the joint-object V6 -> V6_1 discovery chain inside V10.
It does not perform object-native peak discovery, sensitivity testing, pair-order analysis, or physical interpretation.

## Regression audit status
- ruptures_primary_points: pass (key mismatch=0, value diff=0)
- detector_local_peaks_all: pass (key mismatch=0, value diff=0)
- baseline_detected_peaks_registry: pass (key mismatch=0, value diff=0)
- candidate_points_bootstrap_summary: pass (key mismatch=0, value diff=0)
- candidate_points_bootstrap_match_records: pass (key mismatch=0, value diff=0)
- candidate_point_bands: pass (key mismatch=0, value diff=0)
- derived_windows_registry: pass (key mismatch=0, value diff=0)
- window_point_membership: pass (key mismatch=0, value diff=0)
- window_uncertainty_summary: pass (key mismatch=0, value diff=0)
- window_return_day_distribution: pass (key mismatch=0, value diff=0)

## Lineage counts
- strict accepted candidates/members: 4
- non-strict / candidate-lineage rows: 4

## Important interpretation boundary
day-18-like peaks are tracked as joint free-detection candidate lineage when present; they are not automatically contamination and are not automatically strict accepted windows.

## Next intended use
Use `lineage/joint_main_window_lineage_v10_1.csv` as the reference coordinate system before object-native peak discovery or sensitivity tests.