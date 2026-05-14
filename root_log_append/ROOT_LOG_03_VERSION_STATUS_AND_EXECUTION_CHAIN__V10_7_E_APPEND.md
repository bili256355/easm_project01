# ROOT_LOG_03 append — V10.7_e

## V10.7_e — H35 existence attribution audit

- Branch: `stage_partition/V10/v10.7`
- Entry: `scripts/run_h35_existence_attribution_v10_7_e.py`
- Output: `outputs/h35_existence_attribution_v10_7_e`
- Status: implemented as a route-decision audit.
- Scope: H-only, W045/H35 existence attribution.
- Core question: why does H35 appear as an H-only candidate if V10.7_d does not support it as a stable independent event?
- Tested attribution classes:
  - H18-like second stage;
  - seasonal/background or local curvature;
  - few-year-driven feature;
  - method-level score shoulder/background;
  - unresolved possible independent component.
- Not tested here:
  - H35 → W045;
  - H35 → P/V/Je/Jw;
  - E2 multi-object package;
  - causal or physical mechanism.

Primary decision file:

```text
tables/h35_existence_attribution_decision_v10_7_e.csv
```
