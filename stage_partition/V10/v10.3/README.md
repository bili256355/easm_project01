# V10.3 peak discovery sensitivity

This subpackage runs candidate-aware sensitivity tests for the free-season peak discovery workflow.

It covers:

- `joint_all` free-season peak discovery, referenced to V10.1 lineage.
- `P_only`, `V_only`, `H_only`, `Je_only`, `Jw_only` object-native free-season peak discovery, referenced to V10.2 catalogs.

It does **not** perform physical interpretation, pair-order analysis, or re-decision of strict accepted windows.

## Run

```bat
python D:\easm_project01\stage_partition\V10\v10.3\scripts\run_peak_discovery_sensitivity_v10_3.py
```

Debug bootstrap:

```bat
set V10_3_DEBUG_N_BOOTSTRAP=20
python D:\easm_project01\stage_partition\V10\v10.3\scripts\run_peak_discovery_sensitivity_v10_3.py
```

Full bootstrap for all configs can be expensive. Bootstrap mode is controlled by:

```bat
set V10_3_BOOTSTRAP_MODE=baseline_and_match
```

Options:

- `baseline_and_match` (default): bootstrap only for baseline and match-radius variants.
- `all`: bootstrap for every sensitivity config.
- `baseline_only`: bootstrap only for the baseline config.
- `none`: deterministic-only sensitivity.

Number of bootstrap replicates:

```bat
set V10_3_N_BOOTSTRAP=1000
```

## Outputs

All outputs are written inside:

```text
V10\v10.3\outputs\peak_discovery_sensitivity_v10_3
```

Key files:

```text
cross_scope\candidate_peak_sensitivity_by_scope_config_v10_3.csv
cross_scope\bootstrap_support_sensitivity_by_scope_config_v10_3.csv
cross_scope\candidate_band_sensitivity_by_scope_config_v10_3.csv
cross_scope\derived_window_sensitivity_by_scope_config_v10_3.csv
lineage_mapping\candidate_lineage_sensitivity_mapping_v10_3.csv
cross_scope\peak_discovery_sensitivity_summary_by_scope_config_v10_3.csv
PEAK_DISCOVERY_SENSITIVITY_V10_3_SUMMARY.md
```

## Smooth-input sensitivity extension

This patch adds a `smooth_input` single-factor sensitivity group:

```text
SMOOTH_INPUT_5D
```

It compares the baseline smooth9 input against a 5-day smoothed input while keeping the same free-discovery algorithmic flow. The result is reported in the same cross-scope tables with:

```text
sensitivity_group = smooth_input
changed_factor = smoothed_fields_input
changed_value = smooth5
input_source = smooth5
```

Default smooth5 path:

```text
D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a\preprocess\smoothed_fields.npz
```

Override if needed:

```bat
set V10_3_SMOOTH5_FIELDS=D:\path\to\smooth5\smoothed_fields.npz
```

Disable this group if you only want the earlier algorithm-parameter sensitivity:

```bat
set V10_3_INCLUDE_SMOOTHING_GROUP=0
```

The run writes an input inventory here:

```text
cross_scope\input_source_inventory_v10_3.csv
```


### HOTFIX05: Smooth5-internal parameter sensitivity

This extension keeps the existing `SMOOTH_INPUT_5D` comparison, which compares baseline smooth9 against smooth5 input. It also adds `SMOOTH5_*` single-factor configurations that use smooth5 input as the baseline and perturb detector width, detector penalty, peak distance, match radius, candidate-band, and merge parameters inside smooth5.

Important comparison semantics:

- `SMOOTH_INPUT_5D` is compared to `BASELINE`.
- `SMOOTH5_*` configs are compared to `SMOOTH_INPUT_5D`.
- The comparison outputs include `reference_config_id` so downstream review can distinguish smooth9-vs-smooth5 input sensitivity from smooth5-internal algorithm sensitivity.

New summary output:

```text
cross_scope/smooth5_internal_sensitivity_summary_by_scope_factor_v10_3.csv
```

Optional environment flag:

```bat
set V10_3_INCLUDE_SMOOTH5_INTERNAL_GROUP=0
```

to disable smooth5-internal configs while keeping the earlier smooth-input comparison.

## HOTFIX06 closure audit outputs

This patch adds three closure checks requested after the smooth5-internal sensitivity run:

1. `cross_scope/candidate_order_inversion_summary_v10_3.csv` formally audits whether matched candidate peaks invert temporal order under perturbations.
2. `cross_scope/detector_width_bootstrap_support_summary_v10_3.csv` summarizes bootstrap support changes for detector-width configs. In the default `baseline_and_match` mode, detector-width configs now run bootstrap as a minimal closure check.
3. `lineage_mapping/new_candidate_inventory_v10_3.csv` and `cross_scope/candidate_shift_type_summary_v10_3.csv` separate new candidates and explicit shift types from small day drifts.

These are method-layer closure audits only. They do not perform physical interpretation, pair-order analysis, or accepted-window re-decision.
