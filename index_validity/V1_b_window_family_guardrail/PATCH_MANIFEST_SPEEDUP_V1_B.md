# Patch manifest: index_validity V1_b equivalent speedup

Replaces:

- `scripts/run_index_validity_window_family_guardrail_v1_b.py`
- `src/index_validity_v1_b/metrics.py`
- `src/index_validity_v1_b/pipeline.py`

Adds:

- `SPEEDUP_PATCH_NOTES_V1_B.md`

No new scientific version is introduced. This is a performance/diagnostic patch for the existing `V1_b_window_family_guardrail` layer.
