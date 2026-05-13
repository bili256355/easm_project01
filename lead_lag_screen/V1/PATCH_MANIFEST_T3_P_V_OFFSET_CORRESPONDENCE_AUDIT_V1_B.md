# Patch Manifest: T3 P/V850 Offset-Correspondence Audit v1_b

## Purpose

Add an independent object-layer audit that distinguishes:

1. P climatological bands vs P change peaks/bands.
2. V850 climatological structures vs V850 change structures.
3. Same-region comparison vs pre-registered offset/edge/gradient correspondence.

## Added files

```text
scripts/run_t3_p_v_offset_correspondence_audit_v1_b.py
src/lead_lag_screen_v1/t3_p_v_offset_correspondence_settings.py
src/lead_lag_screen_v1/t3_p_v_offset_correspondence_io.py
src/lead_lag_screen_v1/t3_p_v_offset_correspondence_core.py
src/lead_lag_screen_v1/t3_p_v_offset_correspondence_figures.py
src/lead_lag_screen_v1/t3_p_v_offset_correspondence_pipeline.py
README_T3_P_V_OFFSET_CORRESPONDENCE_AUDIT_V1_B.md
PATCH_MANIFEST_T3_P_V_OFFSET_CORRESPONDENCE_AUDIT_V1_B.md
```

## Default command

```bat
python D:\easm_project01\lead_lag_screen\V1\scripts\run_t3_p_v_offset_correspondence_audit_v1_b.py
```

## Notes

- Does not read or compute V->P support/R²/lag/pathway results.
- Does not output full-grid NPZ arrays.
- Uses the same S3/T3/S4 window basis as the current V1/T3 object-layer work.
- All offset candidates are pre-registered fixed offsets or V850 structure positions, not free post-hoc matches.
