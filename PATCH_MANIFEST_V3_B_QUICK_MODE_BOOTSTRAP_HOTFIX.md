# V3_b quick PC1-mode bootstrap hotfix

Default PC1 mode bootstrap is reduced from 500 to 30 and the default output tag is changed to `eof_pc1_smooth5_v3_b_stability_quick_a`. Relation bootstrap remains 1000. Includes prior logger hotfix via `pipeline_stability.py`.

Run:

```bat
cd /d D:\easm_project01
python lead_lag_screen\V3\scripts\run_lead_lag_screen_v3_b_stability.py
```

For 20 mode bootstraps:

```bat
python lead_lag_screen\V3\scripts\run_lead_lag_screen_v3_b_stability.py --mode-bootstrap 20
```

For the heavy formal mode bootstrap later:

```bat
python lead_lag_screen\V3\scripts\run_lead_lag_screen_v3_b_stability.py --formal-mode-bootstrap --output-tag eof_pc1_smooth5_v3_b_stability_formal_a
```
