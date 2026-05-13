# UPDATE_LOG_V7_T

## V7-t: W45 H/Jw raw-field transition-definition audit

### Purpose

This patch adds a method-adjudication branch for W45 H/Jw. It does **not** add lat-band pairing, H-low/Jw-low pairing, component taxonomy, or additional complex relation labels.

The branch audits whether the current pre→post projection progress creates artificial H retreat / H-Jw order ambiguity.

### New entry

```bash
python stage_partition\V7\scripts\run_w45_H_Jw_transition_definition_audit_v7_t.py
```

### New module

```text
stage_partition/V7/src/stage_partition_v7/w45_H_Jw_transition_definition_audit.py
```

### Output directory

```text
stage_partition/V7/outputs/w45_H_Jw_transition_definition_audit_v7_t
```

### Log directory

```text
stage_partition/V7/logs/w45_H_Jw_transition_definition_audit_v7_t
```

### Main diagnostics

The branch computes raw025 two-dimensional field transition diagnostics for H and Jw:

- pre→post projection progress (`P_proj`)
- distance-to-pre/post progress (`P_dist`)
- raw endpoint distances (`D_pre`, `D_post`)
- pattern correlation to pre/post (`R_pre`, `R_post`, `R_diff`)
- orthogonal residual ratio relative to the pre→post axis
- daily change direction cosine relative to the pre→post axis

### Main outputs

```text
w45_H_Jw_transition_metric_curves_v7_t.csv
w45_H_retreat_definition_adjudication_v7_t.csv
w45_H_Jw_order_sync_by_metric_v7_t.csv
w45_H_Jw_definition_consistency_summary_v7_t.csv
w45_H_Jw_method_failure_or_continue_v7_t.csv
w45_H_Jw_transition_definition_audit_summary_v7_t.md
run_meta.json
input_audit_v7_t.json
```

### Interpretation boundary

This branch only audits transition-definition sensitivity. It does not prove causality, H→Jw, Jw→H, physical mechanism, or region-to-region correspondence.

If the definitions conflict, the next step is transition-definition method redesign, not further spatial post-processing.

### Debug options

For quick local checks:

```bash
set V7T_DEBUG_N_BOOTSTRAP=20
set V7T_SKIP_FIGURES=1
python stage_partition\V7\scripts\run_w45_H_Jw_transition_definition_audit_v7_t.py
```

Default bootstrap remains the V7 setting, normally 1000.
