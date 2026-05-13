# V9.1_d EOF transition-mode audit summary

version: `v9_1_d_eof_transition_mode_audit`

## Method boundary
- Read-only relative to V9.
- EOF inputs are whole-window, multi-object year-anomaly profiles, not single-year peak days.
- EOF/PC phases are statistical transition-mode candidates, not physical regime names.
- Peak/order interpretation requires PC phase group reruns of V9 peak logic.

## Configuration
- windows: W045, W081, W113, W160
- n_modes_main: 3
- group_bootstrap_n: 500
- eof_stability_bootstrap_n: 300
- min_group_size_for_peak: 10

## EOF peak/order evidence counts
- not_supported_unstable_eof: 80
- not_supported: 38
- partial_eof_mode_hint: 2

## Mode stability counts
- unstable_mode: 8
- caution_mode: 2
- stable_mode: 2