# V9.1_f_all_pairs_a summary

This is an all-pair coverage extension of V9.1_f. It preserves the bootstrap-composite MCA method and expands the target registry to all 10 P/V/H/Je/Jw object pairs per accepted window.

## Interpretation boundary

- Bootstrap samples are resampled composite perturbations, not independent physical years.
- High/low score groups are bootstrap-space score phases, not physical year types.
- New all-pair results must be interpreted with target coverage and multiple-testing audits.
- This patch does not perform physical interpretation.

## Coverage
- total targets: 40
- original priority targets: 15
- added all-pair targets: 25

## evidence_v3 counts
- robust_common_mode_reversal: 19
- robust_continuous_gradient: 13
- robust_one_sided_locking: 5
- robust_pair_specific_reversal: 3

## Per-window density
- W045: 10 pairs audited; 1 pair-specific, 8 common-mode, 0 locking, 1 gradient labels.
- W081: 10 pairs audited; 1 pair-specific, 2 common-mode, 1 locking, 6 gradient labels.
- W113: 10 pairs audited; 1 pair-specific, 5 common-mode, 3 locking, 1 gradient labels.
- W160: 10 pairs audited; 0 pair-specific, 4 common-mode, 1 locking, 5 gradient labels.
