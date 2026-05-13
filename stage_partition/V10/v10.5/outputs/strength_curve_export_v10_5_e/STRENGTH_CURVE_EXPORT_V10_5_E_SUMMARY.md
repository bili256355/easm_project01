# V10.5_e full-season strength curve export summary

## Scope

This export pulls continuous 4–9 month strength curves from the main-method detector layer and the detector-external profile-energy validation layer.
It does not rerun peak discovery, bootstrap, object-order analysis, accepted-window selection, or physical interpretation.

## Main outputs

- `curves/main_method_continuous_detector_score_curves_v10_5_e.csv`
- `curves/profile_energy_continuous_curves_fullseason_v10_5_e.csv`
- `curves/combined_strength_curves_long_v10_5_e.csv`
- `markers/main_method_candidate_markers_v10_5_e.csv`
- `markers/profile_energy_global_topk_peaks_v10_5_e.csv`
- `figures/main_method_detector_score_curves_fullseason_v10_5_e.png`
- `figures/profile_energy_curves_fullseason_k*_v10_5_e.png`
- `figures/*_main_vs_profile_energy_curves_v10_5_e.png`

## Interpretation boundary

The exported curves expose score landscapes and multi-peak structure. They are diagnostic evidence only. A high score does not imply strict accepted status, physical mechanism, causal order, or year-bootstrap stability.

## Run meta

```json
{
  "status": "success",
  "start_time_utc": "2026-05-13T04:04:05.996463+00:00",
  "end_time_utc": "2026-05-13T04:04:29.084226+00:00",
  "bundle_root": "D:\\easm_project01\\stage_partition\\V10\\v10.5",
  "output_root": "D:\\easm_project01\\stage_partition\\V10\\v10.5\\outputs\\strength_curve_export_v10_5_e",
  "smoothed_fields_path": "D:\\easm_project01\\foundation\\V1\\outputs\\baseline_a\\preprocess\\smoothed_fields.npz",
  "n_main_curve_rows": 930,
  "n_main_marker_rows": 48,
  "n_profile_energy_curve_rows": 2745,
  "n_profile_energy_topk_rows": 143,
  "n_combined_curve_rows": 3675,
  "does_not_rerun_peak_discovery": true,
  "does_not_rerun_bootstrap": true,
  "does_not_redefine_accepted_windows": true,
  "does_not_perform_physical_interpretation": true,
  "settings": {
    "foundation": {
      "project_root": "D:\\easm_project01",
      "foundation_layer": "foundation",
      "foundation_version": "V1",
      "preprocess_output_tag": "baseline_a"
    },
    "profile": {
      "lat_step_deg": 2.0,
      "p_lon_range": [
        105.0,
        125.0
      ],
      "p_lat_range": [
        15.0,
        39.0
      ],
      "v_lon_range": [
        105.0,
        125.0
      ],
      "v_lat_range": [
        10.0,
        30.0
      ],
      "h_lon_range": [
        110.0,
        140.0
      ],
      "h_lat_range": [
        15.0,
        35.0
      ],
      "je_lon_range": [
        120.0,
        150.0
      ],
      "je_lat_range": [
        25.0,
        45.0
      ],
      "jw_lon_range": [
        80.0,
        110.0
      ],
      "jw_lat_range": [
        25.0,
        45.0
      ]
    },
    "curve_export": {
      "k_values": [
        7,
        9,
        11
      ],
      "global_top_k": 15,
      "local_peak_min_distance_days": 3,
      "emit_figures": true
    },
    "output_tag": "strength_curve_export_v10_5_e"
  }
}
```