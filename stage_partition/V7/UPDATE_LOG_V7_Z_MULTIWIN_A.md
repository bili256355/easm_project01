# UPDATE_LOG_V7_Z_MULTIWIN_A

## V7-z-multiwin-a: accepted-window multi-object profile-mainline + 2D mirror

This patch adds a new, isolated multi-window entry point that applies the stabilized W45 clean mainline to all accepted/significant system windows.

Key decisions:

1. `system_window` is the accepted core band from an external accepted-window registry. It is not the detector search range.
2. `detector_search_range` is the wide object-window search range between neighboring accepted windows, with a 5-day right neighbor buffer.
3. `analysis_range` is the C0 pre + system window + C0 post range and is used for daily pre/post curves.
4. `C0/C1/C2` baselines are generated independently for every accepted window.
5. `early/core/late` segments are generated around each `system_window`.
6. Main object-window detection uses raw/profile input only.
7. Pattern remains inside pre-post extraction through `R_diff/S_pattern`.
8. 2D mirror computes the same pre-post metrics on the full object region field, but does not rediscover windows or rewrite final claims by itself.
9. Outputs are written per-window immediately and then summarized cross-window.

This patch depends on the previously delivered V7-z-clean and V7-z-2d-a modules.
