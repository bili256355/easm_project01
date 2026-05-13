# T3 P/V850 Latitudinal Object-Change Audit v1_a

This patch adds an object-layer diagnostic for precipitation and v850 changes across the S3 -> T3 -> S4 sequence.

It intentionally does **not** compute V->P support, R², lag/tau0, pathway, or causal evidence. It only diagnoses how P and V850 fields themselves change.

## New entry

```bat
python D:\easm_project01\lead_lag_screen\V1\scripts\run_t3_p_v_latitudinal_object_change_audit_v1_a.py
```

If cartopy is unavailable:

```bat
python D:\easm_project01\lead_lag_screen\V1\scripts\run_t3_p_v_latitudinal_object_change_audit_v1_a.py --no-cartopy
```

Tables only:

```bat
python D:\easm_project01\lead_lag_screen\V1\scripts\run_t3_p_v_latitudinal_object_change_audit_v1_a.py --no-figures
```

## Default output directory

```text
D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_t3_p_v_latitudinal_object_change_audit_v1_a
```

## Key outputs

Tables:

```text
tables/object_state_region_summary.csv
tables/object_change_region_delta.csv
tables/precip_lat_profile_long.csv
tables/precip_lat_profile_delta_long.csv
tables/v850_lat_profile_long.csv
tables/v850_lat_profile_delta_long.csv
tables/precip_multiband_summary.csv
tables/precip_band_transition_links.csv
tables/precip_latband_integrated_summary.csv
tables/v850_latband_summary.csv
tables/v850_latitudinal_feature_summary.csv
tables/p_v_latitudinal_feature_summary.csv
tables/p_v_object_change_diagnosis_table.csv
```

Figures:

```text
figures/P_mean_state_S3_T3early_T3full_T3late_S4_cartopy.png
figures/V850_mean_state_S3_T3early_T3full_T3late_S4_cartopy.png
figures/P_change_T3full_minus_S3_cartopy.png
figures/P_change_T3late_minus_T3_early_cartopy.png
figures/P_change_S4_minus_T3_full_cartopy.png
figures/V850_change_T3full_minus_S3_cartopy.png
figures/V850_change_T3late_minus_T3_early_cartopy.png
figures/V850_change_S4_minus_T3_full_cartopy.png
figures/P_lat_profile_chain_by_sector.png
figures/P_lat_profile_delta_chain_by_sector.png
figures/V850_lat_profile_chain_by_sector.png
figures/V850_lat_profile_delta_chain_by_sector.png
figures/P_V_object_transition_panel_T3full_minus_S3.png
figures/P_V_object_transition_panel_T3late_minus_T3_early.png
figures/P_V_object_transition_panel_S4_minus_T3_full.png
```

## Default windows

```text
S3       = days 87-106
T3_early = days 107-112
T3_full  = days 107-117
T3_late  = days 113-117
S4       = days 118-154
```

## Scientific boundary

Allowed outputs:

- P mean-state and change maps.
- V850 mean-state and change maps.
- P/V850 latitudinal profile chains.
- P multi-band detection and adjacent-window band matching.
- Latband summaries using the actual available latitude extent.
- A diagnosis table that states object-level changes only, always with reference windows.

Forbidden interpretations from this layer alone:

- V causes P.
- V explains P.
- V->P support increased/decreased.
- Any pathway or mechanism is established.
- A single rain-band northward shift is claimed when multi-band structures are detected.
- North retreat is claimed without checking higher-latitude bands available in the data.
