# T3 V→P Field-Explanation Hard-Evidence Audit v1_a

This output is an independent hard-evidence audit. It does **not** rerun the V1
lead-lag screen and does **not** modify prior T3 physical-hypothesis outputs.

## Purpose

Distinguish whether T3 V→P fixed-index candidate contraction is better explained
as field-level weakening, tau0 replacement, V-component replacement, P-target
regional shift, transition-window mixing, or a local V1 design limitation.

## Window basis

`use_legacy_t3_window = False`

Windows used:

```text
{
  "S3": [
    87,
    106
  ],
  "T3_early": [
    107,
    112
  ],
  "T3_full": [
    107,
    117
  ],
  "T3_late": [
    113,
    117
  ],
  "S4": [
    118,
    154
  ]
}
```

Default is the V6_1 / V1 main-screen basis: S3 87-106, T3 107-117, S4 118-154.
Legacy physical-audit windows are only used when explicitly requested.

## Main evidence layers

1. V-index → P-field positive-lag explained-variance maps.
2. Pre-registered regional response summaries.
3. Positive-lag versus tau0 R² differences.
4. V-component contrast, especially `V_NS_diff - V_strength`.
5. T3 early/full/late and S3/S4 similarity and dilution diagnostics.

## Important interpretation boundaries

- This is not a causal pathway establishment layer.
- Field/region evidence may show stable explanatory structure, but that is not by itself a physical mechanism.
- `v1_design_limitation_candidate` is intentionally left as context-dependent: it requires comparison with old V1 fixed-index results.
- Do not interpret weak T3 maps as complete disappearance of V influence.
- Do not interpret south/SCS response as proof that V controls South China/SCS rainfall.

## Summary

```json
{
  "status": "success",
  "layer_name": "t3_v_to_p_field_explanation_audit_v1_a",
  "input_index_path": "D:\\easm_project01\\foundation\\V1\\outputs\\baseline_smooth5_a\\indices\\index_values_smoothed.csv",
  "input_field_path": "D:\\easm_project01\\foundation\\V1\\outputs\\baseline_smooth5_a\\preprocess\\smoothed_fields.npz",
  "precip_key": "precip_smoothed",
  "window_definition_used": {
    "S3": [
      87,
      106
    ],
    "T3_early": [
      107,
      112
    ],
    "T3_full": [
      107,
      117
    ],
    "T3_late": [
      113,
      117
    ],
    "S4": [
      118,
      154
    ]
  },
  "use_legacy_t3_window": false,
  "day_mapping_mode": "shape183_day_minus_1",
  "year_mapping_mode": "npz_years",
  "n_years_overlap": 45,
  "n_bootstrap": 1000,
  "max_lag": 5,
  "v_indices": [
    "V_strength",
    "V_NS_diff",
    "V_pos_centroid_lat"
  ],
  "regions": {
    "main_meiyu": {
      "lat_min": 24.0,
      "lat_max": 35.0,
      "lon_min": 100.0,
      "lon_max": 125.0
    },
    "south_china": {
      "lat_min": 18.0,
      "lat_max": 25.0,
      "lon_min": 105.0,
      "lon_max": 120.0
    },
    "scs": {
      "lat_min": 10.0,
      "lat_max": 20.0,
      "lon_min": 105.0,
      "lon_max": 130.0
    },
    "south_china_scs": {
      "lat_min": 10.0,
      "lat_max": 25.0,
      "lon_min": 105.0,
      "lon_max": 130.0
    },
    "north_northeast": {
      "lat_min": 35.0,
      "lat_max": 50.0,
      "lon_min": 110.0,
      "lon_max": 135.0
    },
    "main_easm_domain": {
      "lat_min": 10.0,
      "lat_max": 50.0,
      "lon_min": 100.0,
      "lon_max": 135.0
    }
  },
  "n_observed_map_arrays_computed_in_memory": 335,
  "full_grid_map_arrays_persisted": false,
  "figure_plot_backend": "cartopy",
  "n_region_summary_rows": 1290,
  "n_bootstrap_rows": 540,
  "diagnosis_counts": {
    "supported": 3,
    "counter_evidence": 1,
    "mixed_or_weak": 1,
    "requires_context_comparison": 1
  },
  "warnings": [
    "This audit uses smooth5 day-of-season anomalies for variability explanation; it is not an index-validity smoothed-field representativeness test.",
    "Bootstrap uncertainty is computed for regional response series and decision-layer summaries, not full gridpoint maps.",
    "Full-grid map arrays are not persisted as NPZ outputs in this revision.",
    "Cartopy is the default figure backend when available; use --no-cartopy to force plain lon/lat figures.",
    "Old V1 pair results are not used inside hard-evidence map/region calculations."
  ]
}
```
