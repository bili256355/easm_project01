# UPDATE_LOG_V7_Z_MULTIWIN_A_HOTFIX_03

## Scope
Hotfix for V7-z-multiwin-a conservative W45 profile mode.

## Problem fixed
During detector bootstrap, `_select_main_candidate()` was reused on per-bootstrap candidate tables before observed bootstrap support was computed. Those bootstrap candidate tables do not contain `support_class`, causing:

`KeyError: 'support_class'`

## Change
- `_select_main_candidate()` now accepts candidate tables without `support_class`.
- Missing support is marked as `bootstrap_unscored` with neutral support tier.
- Candidate defaults are added defensively for `candidate_id`, `band_start_day`, `band_end_day`, `peak_score`, and `object`.
- System-window relevance and proximity remain the primary bootstrap selection logic.

## Scientific logic
No scientific definitions were changed:
- no change to hardcoded windows;
- no change to detector search range;
- no change to ruptures.Window detector semantics;
- no change to C0/C1/C2 baselines;
- no change to pre-post metrics;
- no change to 2D metrics.
