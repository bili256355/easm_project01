# V1_1 structural V→P screen

## Purpose

`V1_1` is an independent extension route for the V1 lead-lag screen. It does **not** modify V1. It tests whether adding structural P/V indices restores V→P lead-lag support under the same V1-style screen framework.

Main question:

> In the full set of V1 windows, does the T3 V→P contraction come from index projection mismatch rather than disappearance of the P–V correspondence itself?

## Contract

- V1 is read-only.
- V1_1 computes new structural P/V indices itself from `smoothed_fields.npz`.
- V1_1 combines V1 old anomaly indices with new V1_1 structural anomaly indices.
- Main direction is V→P only.
- H / Je / Jw are not included.
- V1 lag tests, AR(1) surrogate max-stat null, audit surrogate null, year-block bootstrap directional robustness, and lag-vs-tau0 stability judgement are retained.
- Outputs are written only under `lead_lag_screen/V1_1/outputs/lead_lag_screen_v1_1_structural_vp_a`.

## Run

```bat
python D:\easm_project01\lead_lag_screen\V1_1\scripts\run_lead_lag_screen_v1_1_structural_vp_a.py
```

Quick connection test:

```bat
python D:\easm_project01\lead_lag_screen\V1_1\scripts\run_lead_lag_screen_v1_1_structural_vp_a.py --debug-fast
```

Formal run:

```bat
python D:\easm_project01\lead_lag_screen\V1_1\scripts\run_lead_lag_screen_v1_1_structural_vp_a.py --n-surrogates 1000 --n-audit-surrogates 1000 --n-direction-bootstrap 1000
```

## New structural V indices

- `V_pos_north_edge_lat`
- `V_pos_south_edge_lat`
- `V_pos_band_width`
- `V_pos_centroid_lat_recomputed`
- `V_highlat_35_55_mean`
- `V_lowlat_20_30_mean`
- `V_high_minus_low_35_55_minus_20_30`
- `V_lowlat_weakening_proxy_20_30`

## New structural P indices

- `P_main_28_35_mean`
- `P_south_10_25_mean`
- `P_scs_10_20_mean`
- `P_highlat_40_60_mean`
- `P_highlat_35_60_mean`
- `P_highlat_minus_main`
- `P_south_minus_main`
- `P_south_plus_highlat_minus_main`

## Key output files

- `indices/v1_1_index_values_raw.csv`
- `indices/v1_1_index_values_doy_anomaly.csv`
- `indices/v1_1_index_registry.csv`
- `indices/v1_1_index_quality_flags.csv`
- `tables/v1_1_v_to_p_lead_lag_long.csv`
- `tables/v1_1_v_to_p_best_positive_lag.csv`
- `tables/v1_1_lag_tau0_stability.csv`
- `tables/v1_1_v_to_p_classified_pairs.csv`
- `tables/v1_vs_v1_1_pair_recovery_summary.csv`
- `tables/v1_1_new_index_contribution_by_window.csv`
- `summary/summary.json`
- `summary/run_meta.json`

## Interpretation boundary

V1_1 answers whether structural indices recover V→P temporal eligibility. It does not establish causality or a physical pathway.
