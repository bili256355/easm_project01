# UPDATE_LOG_V7_Z_MULTIWIN_A_HOTFIX_06

## Version

`V7-z-multiwin-a hotfix_06`: W45 profile-only peak/order + pre-post curve-order tests.

## Scope

This hotfix does **not** modify the restored hotfix05 detector, window ranges, C0/C1/C2 baseline rules, 2D mirror definitions, or all-window execution logic.  It only adds W45 profile-only order-test outputs.

## Added diagnostics

- Empirical timing-resolution audit from bootstrap selected peak return days.
- Pairwise peak-order tests.
- Pairwise synchrony / equivalence tests using data-derived `tau_sync`.
- Pairwise selected-window overlap tests.
- Pairwise profile state-progress difference tests based on `ΔS_AB(t) = S_A(t) - S_B(t)` functionals, not day-by-day significance tests.
- Pairwise state catch-up / reversal observed summaries.
- Object-level growth sign structure, including positive/negative growth areas and negative-growth support.
- Object-level growth pulse / multi-stage structure, focused on growth mass distribution rather than exact episode-day stability.
- Pairwise growth process difference tests.
- Pairwise pre-post curve interpretation and combined order interpretation tables.

## Important interpretation constraints

- `peak_order_unresolved` is not synchrony.
- `window_overlap_strong` is not synchrony.
- `synchrony_supported` is only assigned when the pairwise peak-delta bootstrap interval falls inside the empirical detector timing-resolution range.
- Pre-post curve evidence is layer-specific state/growth evidence and must not be converted directly into hard peak lead.
