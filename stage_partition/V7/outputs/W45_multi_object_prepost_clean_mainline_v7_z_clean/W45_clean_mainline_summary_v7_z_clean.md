# W45 multi-object pre-post clean mainline (V7-z-clean)

## Purpose
This clean mainline keeps raw/profile as the only object-window detection input, and keeps pattern inside pre-post extraction via R_diff / S_pattern.

## Object-window selected candidates
- H: peak day 31, window day 21–39, support=0.821, class=candidate_window_80
- Je: peak day 46, window day 44–48, support=0.759, class=weak_window_50
- Jw: peak day 15, window day 12–19, support=0.986, class=accepted_window_95
- P: peak day 43, window day 33–49, support=0.976, class=accepted_window_95
- V: peak day 44, window day 37–51, support=0.995, class=accepted_window_95

## Final claims
- H-P: co_transition_with_A_curve_tendency (Level2_curve_tendency_only)
  - Allowed: A and B co-transition in raw/profile object-window; A shows curve-level ahead tendency.
  - Forbidden: Do not write A leads B as a hard object-window timing claim.
- Je-P: co_transition_with_A_curve_tendency (Level3_curve_supported_with_cotransition)
  - Allowed: A and B co-transition in raw/profile object-window; A shows curve-level ahead tendency.
  - Forbidden: Do not write A leads B as a hard object-window timing claim.
- Je-V: co_transition_with_A_curve_tendency (Level3_curve_supported_with_cotransition)
  - Allowed: A and B co-transition in raw/profile object-window; A shows curve-level ahead tendency.
  - Forbidden: Do not write A leads B as a hard object-window timing claim.
- P-V: co_transition_with_A_curve_tendency (Level2_curve_tendency_only)
  - Allowed: A and B co-transition in raw/profile object-window; A shows curve-level ahead tendency.
  - Forbidden: Do not write A leads B as a hard object-window timing claim.

## Downgraded signals
- H-Je: B_curve_tendency_only (Do not promote to lead without raw/profile window support.)
- H-Jw: A_curve_tendency_only (Do not promote to lead without raw/profile window support.)
- H-V: A_curve_tendency_only (Do not promote to lead without raw/profile window support.)
- Je-Jw: A_curve_tendency_only (Do not promote to lead without raw/profile window support.)
- Jw-P: B_curve_tendency_only (Do not promote to lead without raw/profile window support.)
- Jw-V: A_curve_tendency_only (Do not promote to lead without raw/profile window support.)

## Method notes
- Shape-normalized pattern detector is not used as a main object-window input in this clean mainline.
- R_diff / S_pattern remains a pre-post pattern-similarity extraction branch.
- Co-transition veto prevents multiple weak curve tendencies from becoming hard lead claims.