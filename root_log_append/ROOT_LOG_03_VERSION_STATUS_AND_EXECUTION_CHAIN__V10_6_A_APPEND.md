# Root log append: V10.6_a version status and execution chain

## V10.6_a — W045 precluster audit

- **Branch**: `stage_partition/V10/v10.6`
- **Entry script**: `stage_partition/V10/v10.6/scripts/run_w045_precluster_audit_v10_6_a.py`
- **Output directory**: `stage_partition/V10/v10.6/outputs/w045_precluster_audit_v10_6_a`
- **Engineering state**: implemented as a new V10.6 branch; run status must be checked locally after execution.
- **Research role**: method-layer / derived-structure audit for W045 only.
- **Main question**: decompose W045 neighborhood into E1 day16–19, E2 day30–35, M day41–46, and H_post day57 reference; audit whether H35 can be treated as confirmed weak precursor.
- **Inputs**: V10.5_e strength curves and candidate markers; V10.5_b candidate-family summary; V10.5_d key competition cases.
- **Does not do**: accepted-window re-detection; yearwise validation; cartopy/spatial validation; causal inference; full-season generalization.
- **Interpretation boundary**: V10.6_a cannot prove H35 as confirmed weak precursor. It only classifies H35's method-layer role and records evidence against over-strong interpretation.
