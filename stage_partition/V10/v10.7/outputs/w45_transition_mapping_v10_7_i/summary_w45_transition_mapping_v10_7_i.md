# V10.7_i W33→W45 cross-object transition mapping audit

## Method boundary
- This audit does not test same-object E2–M similarity; V10.7_h already handled that.
- This audit does not control away P/V/Je/Jw as covariates.
- It asks whether the E2/W33 object vector maps to the M/W45 object vector, allowing cross-object reorganization.
- It allows H_E2 to map to non-H M objects such as M_Jw, M_P, or M_V.
- It is not causal inference.

## Route decision
- **E2_to_M_cross_object_mapping**: `no_mapping`. Evidence: cv_r2=-0.181, p=0.448, null_p90=-0.117. Implication: No scalar cross-object mapping detected; W33 should not yet be treated as W45 preconfiguration.
- **H_E2_contribution_to_E2_M_mapping**: `negative_or_unstable_dimension`. Evidence: skill_drop=-0.008, p=0.159, null_p90=-0.001. Implication: Do not interpret H contribution because overall E2-to-M mapping is not established.
- **H_E2_target_specific_mapping**: `H_E2_maps_to_M_target`. Evidence: M_Je:weak_mapping_support,r=-0.29,p=0.065. Implication: If H_E2 maps to non-H M targets, retain H as preconfiguration candidate; otherwise H remains unresolved/secondary.
- **W33_to_W45_route**: `W33_not_connected_to_W45_by_this_scalar_mapping`. Evidence: mapping=no_mapping; H_contribution=negative_or_unstable_dimension; H_target_mapping=H_E2_maps_to_M_target. Implication: W18-W33 sequence may be pre-window activity not connected to W45 under scalar mapping; consider richer position/shape metrics before final rejection.

## Forbidden interpretations
- Do not interpret pairwise or multivariate mapping as causality.
- Do not require H_E2 to map to H_M; H may matter through non-H M targets.
- Do not use this scalar mapping to reject position/shape-based H roles; it only tests strength proxies.