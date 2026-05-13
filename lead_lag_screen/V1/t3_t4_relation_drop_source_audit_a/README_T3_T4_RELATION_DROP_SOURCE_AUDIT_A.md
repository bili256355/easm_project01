# V1 T3/T4 relation drop source audit A

This folder is an independent V1 audit layer. It does **not** modify V1 screening code or V1 existing results.

## Purpose

The audit localizes why T3 and T4 have very low usable lead-lag relationship counts in the V1 stability-judged output.

It reads the V1 post-processed stability judgement table and outputs:

- gate funnel by window;
- S3/T3/S4/T4/S5 pair survival chain;
- T3/T4 family-direction gap contribution;
- T3/T4 near-miss rows;
- effect-size distribution summaries;
- old tier to new stability judgement transition;
- diagnosis table.

## Default input

```bat
D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b\tables\lead_lag_pair_summary_stability_judged.csv
```

## Default output

```bat
D:\easm_project01\lead_lag_screen\V1\t3_t4_relation_drop_source_audit_a\outputs
```

## Run

```bat
python D:\easm_project01\lead_lag_screen\V1\t3_t4_relation_drop_source_audit_a\run_v1_t3_t4_relation_drop_source_audit_a.py
```

Optional explicit paths:

```bat
python D:\easm_project01\lead_lag_screen\V1\t3_t4_relation_drop_source_audit_a\run_v1_t3_t4_relation_drop_source_audit_a.py ^
  --project-root D:\easm_project01 ^
  --input-csv D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b\tables\lead_lag_pair_summary_stability_judged.csv
```

## Main outputs

```text
tables/window_gate_funnel_summary.csv
tables/family_direction_T3_T4_gap_contribution.csv
tables/pair_survival_chain_S3_T3_S4_T4_S5.csv
tables/T3_T4_near_miss_pairs.csv
tables/window_effect_size_distribution_summary.csv
tables/tier_to_stability_transition_by_window.csv
tables/failure_reason_counts_by_window.csv
tables/t3_t4_drop_source_diagnosis_table.csv
summary/summary.json
summary/run_meta.json
logs/RUN_LOG.md
```

## Interpretation boundary

This audit can identify whether T3/T4 are mostly filtered before positive-lag support, by old-tier collapse, by family-direction narrowing, by tau0/reverse filters, or by near-miss failures.

It does **not** prove the physical cause of T3/T4 weakening. Treat its diagnosis as a gate-localization result, not a mechanism result.
