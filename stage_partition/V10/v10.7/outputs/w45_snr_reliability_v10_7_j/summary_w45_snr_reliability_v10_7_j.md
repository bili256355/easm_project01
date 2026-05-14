# V10.7_j W45 E2–M yearwise SNR / reliability audit

## Method boundary

- This is not a new W33→W45 mechanism test.
- It audits whether V10.7_i's yearwise scalar indicators are reliable enough for mapping tests.
- If reliability is low, V10.7_i non-detection should be treated as non-decisive rather than as a structural negative result.

## Key route decisions

- **E2_M_yearwise_indicator_snr**: `sufficient_indicator_snr` — usable=19/20; usable_or_marginal=20/20. V10.7_i scalar mapping non-detection is more interpretable, though still limited to scalar indicators.
- **leave_days_out_mapping_sensitivity**: `leave_day_stable` — stable_pair_fraction=1.000. Mapping signs are not dominated by individual days for most pairs.
- **window_shift_mapping_sensitivity**: `window_shift_stable` — stable_pair_fraction=0.940. Mapping signs are relatively robust to +/-2 day shifts for most pairs.
- **object_usability_P**: `usable_object_for_yearwise_mapping` — usable=4/4, marginal=0/4 among primary E2/M modes. P scalar indicators can support yearwise mapping checks.
- **object_usability_V**: `usable_object_for_yearwise_mapping` — usable=4/4, marginal=0/4 among primary E2/M modes. V scalar indicators can support yearwise mapping checks.
- **object_usability_H**: `usable_object_for_yearwise_mapping` — usable=3/4, marginal=1/4 among primary E2/M modes. H scalar indicators can support yearwise mapping checks.
- **object_usability_Je**: `usable_object_for_yearwise_mapping` — usable=4/4, marginal=0/4 among primary E2/M modes. Je scalar indicators can support yearwise mapping checks.
- **object_usability_Jw**: `usable_object_for_yearwise_mapping` — usable=4/4, marginal=0/4 among primary E2/M modes. Jw scalar indicators can support yearwise mapping checks.

## Interpretation guardrails

- Low SNR does not prove absence of E2→M organization; it means the current yearwise scalar mapping has low power.
- Good SNR still only supports scalar-indicator mapping, not causality or shape/position mechanisms.
- Window-shift or leave-day sensitivity indicates fixed-window timing may be too rigid.
