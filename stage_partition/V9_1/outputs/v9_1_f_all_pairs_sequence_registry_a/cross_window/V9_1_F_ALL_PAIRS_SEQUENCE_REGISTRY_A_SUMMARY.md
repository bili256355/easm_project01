# V9.1_f all-pairs sequence registry A

This is a read-only derived registry. It does not rerun bootstrap, MCA/SVD, or the peak detector.

## Purpose

For each V9.1_f_all_pairs target, use the target score high/mid/low groups and the precomputed bootstrap object peaks to summarize the full five-object peak sequence under each score phase.

## Key boundaries

- The sequence is target-conditioned: each target has its own score direction and therefore its own high/low grouping.

- These outputs are not physical regimes and not independent year types.

- Results are derived from bootstrap-composite samples, not direct-year samples.

- This layer exposes window-level sequence information that was already latent in the bootstrap peaks but not previously summarized.

## Output overview

- targets summarized: 40

- windows summarized: 4


## Window summary

| window_id   |   n_targets_summarized |   n_original_priority_targets |   n_added_all_pair_targets |   mean_n_pairs_order_changed_high_low |   max_n_pairs_order_changed_high_low |   mean_max_abs_object_shift |   max_abs_object_shift | summary_scope                                                                       |
|:------------|-----------------------:|------------------------------:|---------------------------:|--------------------------------------:|-------------------------------------:|----------------------------:|-----------------------:|:------------------------------------------------------------------------------------|
| W045        |                     10 |                             3 |                          7 |                                   1.6 |                                    3 |                        3.6  |                      6 | target-conditioned high-low window-level peak sequence, not physical interpretation |
| W081        |                     10 |                             3 |                          7 |                                   3.3 |                                    4 |                        4.1  |                      5 | target-conditioned high-low window-level peak sequence, not physical interpretation |
| W113        |                     10 |                             5 |                          5 |                                   1.8 |                                    3 |                        3.9  |                      5 | target-conditioned high-low window-level peak sequence, not physical interpretation |
| W160        |                     10 |                             4 |                          6 |                                   1.8 |                                    3 |                        4.05 |                      7 | target-conditioned high-low window-level peak sequence, not physical interpretation |



## Run metadata

```json

{
  "created_at": "2026-05-11T09:58:34",
  "does_not_rerun_bootstrap": true,
  "does_not_rerun_mca_or_svd": true,
  "does_not_rerun_peak_detector": true,
  "input_dir": "D:\\easm_project01\\stage_partition\\V9_1\\outputs\\bootstrap_composite_mca_audit_v9_1_f_all_pairs_a",
  "input_tag": "bootstrap_composite_mca_audit_v9_1_f_all_pairs_a",
  "long_bootstrap_object_peak_table_saved": false,
  "n_object_peak_stat_rows": 600,
  "n_pair_order_rows": 1200,
  "n_sequence_rows": 120,
  "n_targets": 40,
  "n_windows": 4,
  "near_day_threshold": 0.0,
  "object_shift_report_threshold_days": 3.0,
  "output_dir": "D:\\easm_project01\\stage_partition\\V9_1\\outputs\\v9_1_f_all_pairs_sequence_registry_a",
  "output_tag": "v9_1_f_all_pairs_sequence_registry_a",
  "physical_interpretation_included": false,
  "sequence_scope": "target-conditioned high/mid/low score phases using precomputed bootstrap object peaks",
  "version": "v9_1_f_all_pairs_sequence_registry_a"
}

```
