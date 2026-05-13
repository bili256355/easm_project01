# UPDATE_LOG_V7_R

## w45_H_Jw_feature_relation_exploration_v7_r

Purpose: reduce the current W45 feature-level investigation to the H/Jw pair and mine the feature/component-level information that V7-q exposed.

Scope:
- Diagnostic branch, not a clean trunk rebuild.
- Reads V7-q feature-level outputs as diagnostic input.
- Does not read V7-m/V7-n/V7-o/V7-p outputs as input.
- Does not claim global H leads Jw.
- Does not translate feature IDs into physical regions unless provenance permits it.

Main outputs:
- H/Jw feature provenance audit.
- H/Jw feature eligibility audit.
- H/Jw departure same-phase evidence.
- H early-progress support vs Jw.
- Jw catch-up / finish support vs H.
- H/Jw distribution overlap by phase.
- H/Jw weighted vs unweighted contrast.
- H/Jw feature-coordinate concentration audit.
- H/Jw special relation card and summary.

Interpretation guardrails:
- Near-same-phase is not synchrony.
- H early-progress support is not global H→Jw lead.
- Feature-index concentration is not physical-region evidence without metadata/provenance.
- Weak/noisy/low-contribution features are retained and flagged, not deleted.
