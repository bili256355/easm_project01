# T3 P/V850 Offset-Correspondence Audit v1_b

This output is an object-layer offset-correspondence audit. It separates:

- `P_clim_band`: peaks/bands on window-mean precipitation profiles.
- `P_change_peak` / `P_change_band`: positive/negative centers on window-difference precipitation profiles.
- `V_clim_structure`: V850 positive peak/centroid/edges on window-mean V850 profiles.
- `V_change_structure`: V850 change peak/trough/gradient and V850 positive-edge shifts.

It does not compute V->P support, R2, lag/tau0, pathway, or causality.

The central purpose is to avoid same-region same-sign overinterpretation and to test whether P object changes are more consistent with pre-registered V850 offset, edge, or gradient structures.
