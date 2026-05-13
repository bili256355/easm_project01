# lead_lag_screen/V3: smooth5 field EOF-PC1 lead-lag audit

## Purpose

This layer audits whether V1 smooth5 index-based lead-lag weakening around T3 / the Meiyu-ending window is caused by index applicability limits rather than actual field-to-field relation collapse.

It replaces the 20 manual indices with one window-wise EOF PC1 per physical object:

- P_PC1
- V_PC1
- H_PC1
- Je_PC1
- Jw_PC1

It then reuses the V1 lead-lag semantics as closely as possible:

- smooth5 anomaly field input;
- target-side windows;
- positive lags +1..+5;
- lag0 as same-day diagnostic;
- AR(1) surrogate max-stat null;
- audit surrogate null;
- year-bootstrap directional robustness.

## Non-goals

V3 does **not** do pathway reconstruction, PCMCI, causal discovery, or PC2/PC3 analysis. It is a PC1-only field-mode audit.

## Run

From the project root:

```bat
cd /d D:\easm_project01
python lead_lag_screen\V3\scripts\run_lead_lag_screen_v3.py
```

## Default output

```text
D:\easm_project01\lead_lag_screen\V3\outputs\eof_pc1_smooth5_v3_a
```

Key outputs:

- `eof_pc1_quality.csv`
- `eof_pc1_scores_long.csv`
- `eof_pc1_loadings_long.csv`
- `eof_pc1_loadings.npz`
- `eof_pc1_pair_summary.csv`
- `eof_pc1_family_rollup.csv`
- `v1_index_vs_v3_pc1_family_comparison.csv`
- `t3_meiyu_end_pc1_audit.csv`
- `summary.json`
- `run_meta.json`

## Interpretation guardrails

- V3 strong while V1 weak means index applicability is a plausible source of V1 weakening.
- V3 weak with high PC1 quality suggests main field-mode relation weakening, not necessarily all field relations disappearing.
- V3 weak with low PC1 quality means PC1 is insufficient; do not call the relation absent.
- Lagged weak but lag0 strong indicates same-day/fast adjustment rather than relation disappearance.
