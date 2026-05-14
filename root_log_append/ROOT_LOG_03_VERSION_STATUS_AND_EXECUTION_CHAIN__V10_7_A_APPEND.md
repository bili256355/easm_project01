# ROOT_LOG_03 append: V10.7_a

## V10.7_a — H-only main-method event atlas

- **Version path**: `stage_partition/V10/v10.7`
- **Entry**: `stage_partition/V10/v10.7/scripts/run_h_object_event_atlas_v10_7_a.py`
- **Output**: `stage_partition/V10/v10.7/outputs/h_object_event_atlas_v10_7_a`
- **Role**: H-only main-method baseline export with detector_width sensitivity.
- **Default detector_width grid**: `12, 16, 20, 24, 28`.
- **Baseline width**: `20`.
- **Expected inherited H width=20 candidates**: `19, 35, 57, 77, 95, 115, 129, 155`.
- **Engineering status**: implemented as a new independent V10.7 entry and output directory.
- **Interpretation status**: method-layer baseline only; not a physical or causal interpretation layer.
- **Not implemented in V10.7_a**: bootstrap recurrence, yearwise validation, cartopy spatial continuity validation, multi-object expansion, causal inference.

V10.7_a should be used to choose H event packages for later tests, not to make final claims about H precursor/condition roles.
