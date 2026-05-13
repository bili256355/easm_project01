# Patch manifest: EOF-PC1 V1 adjudication audit A

## Added files

```text
lead_lag_screen/V1_1/scripts/run_v1_1_eof_pc1_v1_adjudication_a.py
lead_lag_screen/V1_1/src/lead_lag_screen_v1_1/eof_pc1_v1_adjudication_settings.py
lead_lag_screen/V1_1/src/lead_lag_screen_v1_1/eof_pc1_v1_adjudication_core.py
lead_lag_screen/V1_1/src/lead_lag_screen_v1_1/eof_pc1_v1_adjudication_pipeline.py
lead_lag_screen/V1_1/README_EOF_PC1_V1_ADJUDICATION_A.md
lead_lag_screen/V1_1/patch_manifests/PATCH_MANIFEST_EOF_PC1_V1_ADJUDICATION_A.md
```

## Scope

This patch only adds an audit layer inside V1_1. It does not modify V1, does not modify the V1_1 structural VP main screen, and does not use the V1_1 high-latitude branch as the adjudication criterion.

## Purpose

Determine whether EOF-PC1 is eligible to adjudicate the V1 old-index T3 V→P weakening by checking:

1. PC1 alignment with old V/P index spaces.
2. Old-index aggregate PC1 lead-lag behavior.
3. EOF-PC1 sensitivity to seasonal/background controls.
4. EOF-PC1 V1-style lag-vs-tau0 classification.
