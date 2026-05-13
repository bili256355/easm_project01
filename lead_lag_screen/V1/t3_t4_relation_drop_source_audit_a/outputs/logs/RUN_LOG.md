# V1 T3/T4 relation drop source audit log

Input: `D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b\tables\lead_lag_pair_summary_stability_judged.csv`
Output: `D:\easm_project01\lead_lag_screen\V1\t3_t4_relation_drop_source_audit_a\outputs`

This audit is read-only with respect to V1 main outputs. It localizes where T3/T4 relationship-count collapse occurs by gate funnel, pair survival, family-direction gap, near-miss rows, and effect-size distributions.

Key caution: this audit localizes *where* pairs are filtered or lost. It does not by itself prove the physical or statistical cause of the drop.