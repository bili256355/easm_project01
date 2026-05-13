# UPDATE_LOG_V7_Z_JE_AUDIT_C

## V7-z-je-audit-c: Je physical variance audit for W45

Purpose: test whether the Je day30-34 low shape-norm episode can be interpreted physically as a low spatial-variance / flattened-profile episode.

Scope:
- Adds a standalone Je-only audit script and module.
- Does not modify V7-z main workflow, detectors, evidence gates, or previous Je audit outputs.
- Reads smoothed u200 fields and rebuilds Je profile / 2D regional metrics.

Key outputs:
- `Je_physical_variance_timeseries_v7_z_c.csv`
- `Je_physical_variance_window_summary_v7_z_c.csv`
- `Je_physical_variance_bootstrap_summary_v7_z_c.csv`
- `Je_physical_variance_minimum_bootstrap_summary_v7_z_c.csv`
- `Je_physical_variance_decision_v7_z_c.csv`
- `Je_physical_variance_audit_summary_v7_z_c.md`

Interpretation boundary:
- This audit can support a low-spatial-variance / flattened-profile interpretation of Je day30-34.
- It cannot establish causality or upgrade Je day33 to the same level as the robust raw/profile day46 main adjustment.
