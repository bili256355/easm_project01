# V10.5_e full-season strength curve export

This add-on exports continuous 4–9 month strength/score curves from both the main-method detector layer and the detector-external profile-energy layer.

Run:

```bat
python D:\easm_project01\stage_partition\V10\v10.5\scripts\run_strength_curve_export_v10_5_e.py
```

Optional foundation override:

```bat
set V10_5_SMOOTHED_FIELDS=D:\your_path\smoothed_fields.npz
python D:\easm_project01\stage_partition\V10\v10.5\scripts\run_strength_curve_export_v10_5_e.py
```

Main outputs:

- `outputs/strength_curve_export_v10_5_e/curves/main_method_continuous_detector_score_curves_v10_5_e.csv`
- `outputs/strength_curve_export_v10_5_e/curves/profile_energy_continuous_curves_fullseason_v10_5_e.csv`
- `outputs/strength_curve_export_v10_5_e/curves/combined_strength_curves_long_v10_5_e.csv`
- `outputs/strength_curve_export_v10_5_e/markers/main_method_candidate_markers_v10_5_e.csv`
- `outputs/strength_curve_export_v10_5_e/markers/profile_energy_global_topk_peaks_v10_5_e.csv`

Boundary: this script exports curves only. It does not rerun peak discovery, bootstrap, order analysis, physical interpretation, or accepted-window selection.
