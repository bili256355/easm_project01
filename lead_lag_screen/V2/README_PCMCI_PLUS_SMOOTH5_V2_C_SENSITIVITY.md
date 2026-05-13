# lead_lag_screen/V2_c targeted PCMCI+ sensitivity patch

This patch replaces the expensive all-window C1 plan with a targeted diagnostic:

- C1 `c1_targeted_no_same_family_controls`: run only for `V -> P` in `S3` and `T3`.
- C2 `c2_family_representative_standard`: run the representative-variable PCMCI+ sensitivity on all 9 windows.

The purpose is to test the specific failure mode exposed by V2_a/V2_b: V->P nearly disappears, including in the Meiyu-season window. The outputs remain sensitivity diagnostics, not pathway results and not final scientific conclusions.

## Run

```bat
cd /d D:\easm_project01
python lead_lag_screen\V2\scripts\run_lead_lag_screen_v2_c_sensitivity.py
```

## Output

```text
D:\easm_project01\lead_lag_screen\V2\outputs\pcmci_plus_smooth5_v2_c_s3_t3_vp_c2_all_a
```

## Interpretation

- If C1 restores V->P in S3/T3, V2_a likely over-conditioned the V/P same-field derived-index structure.
- If C2 restores family-level V->P across windows, the full 20-index PCMCI+ pool likely caused graph-selection collapse.
- If both C1 and C2 still lose V->P, PCMCI+ in this window/index design should not be treated as a reliable control without method redesign.
