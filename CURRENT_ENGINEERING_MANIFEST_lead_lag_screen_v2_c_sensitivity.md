# CURRENT_ENGINEERING_MANIFEST: lead_lag_screen/V2_c targeted PCMCI+ sensitivity

## Patch role

This patch modifies the previous V2_c sensitivity plan to reduce runtime and target the core failure:

- C1 no-same-family-controls is restricted to `V -> P` in `S3` and `T3`.
- C2 representative-variable PCMCI+ still runs on all 9 windows.

## New/updated entry

```text
lead_lag_screen/V2/scripts/run_lead_lag_screen_v2_c_sensitivity.py
```

## Output directory

```text
lead_lag_screen/V2/outputs/pcmci_plus_smooth5_v2_c_s3_t3_vp_c2_all_a
```

## Main boundaries

- This is not a pathway layer.
- This does not overwrite V2_a or V2_b_audit.
- This does not claim PCMCI+ scientific validity; it tests why V2_a loses V->P.
- C1 is targeted, not all-window.
- C2 is all-window but representative-pool only.
