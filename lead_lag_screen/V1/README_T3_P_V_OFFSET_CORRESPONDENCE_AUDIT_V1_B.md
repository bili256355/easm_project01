# T3 P/V850 Offset-Correspondence Audit v1_b

This patch adds an independent object-layer **P/V850 offset-correspondence audit**.

It separates precipitation climatological bands from precipitation change peaks/bands:

- `P_clim_band`: a peak/band on a window-mean precipitation profile.
- `P_change_peak` / `P_change_band`: a positive or negative peak/band on a window-difference precipitation profile.
- `V_clim_structure`: V850 positive peak, centroid, north edge, and south edge on a window-mean V850 profile.
- `V_change_structure`: V850 change peak/trough/gradient and V850 positive-edge shifts.

This layer does **not** compute V->P support, R², lag/tau0, pathway, or causality. It only checks whether P object changes are more consistent with pre-registered V850 offset/edge/gradient structures rather than same-region same-sign comparison.

## New entry

```bat
python D:\easm_project01\lead_lag_screen\V1\scripts\run_t3_p_v_offset_correspondence_audit_v1_b.py
```

Tables only:

```bat
python D:\easm_project01\lead_lag_screen\V1\scripts\run_t3_p_v_offset_correspondence_audit_v1_b.py --no-figures
```

## Default output directory

```text
D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_t3_p_v_offset_correspondence_audit_v1_b
```

## Key tables

```text
tables/p_lat_profile_long.csv
tables/v850_lat_profile_long.csv
tables/p_lat_profile_delta_long.csv
tables/v850_lat_profile_delta_long.csv
tables/p_clim_band_summary.csv
tables/p_change_band_summary.csv
tables/v_clim_structure_summary.csv
tables/v_change_structure_summary.csv
tables/p_clim_band_to_v_clim_structure_summary.csv
tables/p_change_peak_to_v_change_structure_summary.csv
tables/p_highlat_v_north_edge_correspondence.csv
tables/p_v_offset_correspondence_diagnosis_table.csv
```

## Key figures

```text
figures/P_clim_bands_vs_V_clim_structure_chain.png
figures/P_change_peaks_vs_V_change_structure_T3_full_minus_S3.png
figures/P_change_peaks_vs_V_change_structure_T3_late_minus_T3_early.png
figures/P_change_peaks_vs_V_change_structure_S4_minus_T3_full.png
figures/P_highlat_V_north_edge_chain.png
figures/P_south_retention_vs_V_retreat_chain.png
```

## Interpretation boundary

Allowed: object-layer correspondence statements such as “V850 positive north-edge retreat is object-layer consistent with high-latitude P decline.”

Forbidden: causal or pathway statements such as “V850 caused precipitation retreat” or “V->P pathway is established.”
