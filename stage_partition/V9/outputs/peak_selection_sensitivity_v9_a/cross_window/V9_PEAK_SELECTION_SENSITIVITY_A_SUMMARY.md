# V9 peak-selection sensitivity A summary

version: `V9_peak_selection_sensitivity_a`

## Purpose
This run fixes the V7/V9 accepted main windows and perturbs only the object-peak selection layer.
It tests whether P/V/H/Je/Jw selected peak days, peak bands, pairwise order, and five-object sequences are stable under detector-scale, search-range, selection-rule, and smoothing-scale changes.

## Fixed elements
- Main windows are fixed: W045, W081, W113, W160.
- Object definitions are unchanged from V9.
- No changepoint detection is rerun.
- No physical mechanism interpretation is included.

## Perturbation grid
- smoothing: smooth9, smooth5
- detector scales: 16/8, 20/10, 24/12
- search modes: narrow_search, baseline_search, wide_search
- selection rules: baseline_rule, max_score, closest_anchor, max_overlap

## Baseline reproduction
- matched V9 original selected peak day: 20/20

## Window sequence stability
- W045: sequence_highly_sensitive; most_common=Je -> H -> P/V -> Jw (freq=0.21); n_unique=22
- W081: sequence_highly_sensitive; most_common=H -> Je/P/Jw/V (freq=0.15); n_unique=15
- W113: sequence_highly_sensitive; most_common=Je/Jw -> H/P/V (freq=0.25); n_unique=13
- W160: sequence_highly_sensitive; most_common=P -> V/Je/Jw -> H (freq=0.12); n_unique=13

## Most sensitive object peaks
- W045 Je: range=48.0, class=peak_day_rule_sensitive, smooth_delta=2.0
- W045 Jw: range=33.0, class=peak_day_rule_sensitive, smooth_delta=0.0
- W045 H: range=31.0, class=peak_day_rule_sensitive, smooth_delta=4.5
- W045 V: range=31.0, class=peak_day_rule_sensitive, smooth_delta=-1.0
- W081 Je: range=28.0, class=peak_day_rule_sensitive, smooth_delta=2.0
- W113 H: range=27.0, class=peak_day_rule_sensitive, smooth_delta=0.0
- W113 P: range=25.0, class=peak_day_rule_sensitive, smooth_delta=1.5
- W160 Jw: range=20.0, class=peak_day_rule_sensitive, smooth_delta=0.0
- W160 H: range=15.0, class=peak_day_rule_sensitive, smooth_delta=0.0
- W113 Jw: range=12.0, class=peak_day_rule_sensitive, smooth_delta=-1.0

## Interpretation boundary
- These outputs are sensitivity diagnostics only.
- A stable peak day can be used as a V9 reference peak; a sensitive peak should be downgraded.
- Selected peak bands should not be interpreted as physical sub-windows unless band stability is also supported.
