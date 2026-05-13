# UPDATE_LOG_V7_Z_MULTIWIN_A_HOTFIX_01

## Purpose

Conservative hotfix after W045 regression failure in the first multiwin-a run.

## Root cause fixed

The earlier multiwin-a implementation reused a clean-mainline left/right mean-distance detector. That detector was not equivalent to the original V7-z W45 `ruptures.Window` detector and caused W045 H/Jw candidate dates to jump earlier.

## Changes

1. Restored original V7-z `ruptures.Window` semantics for profile object-window detection.
2. Changed default run to W45-only, profile-only:
   - `window_mode = w45`
   - `run_2d = False`
3. Kept ability to run all windows by setting `V7_MULTI_WINDOW_MODE=all`.
4. Kept ability to run 2D mirror by setting `V7_MULTI_RUN_2D=1`.
5. Added run-window selection audit tables.
6. Hardened main candidate selection:
   - system-window-relevant candidates are considered first;
   - far-pre/far-late peaks are retained as secondary candidates, not current-window main by default.
7. Observed and bootstrap selected peaks now use the same selection routine.

## Not changed

- No new scientific metric was added.
- No pre/post baseline definition was changed.
- No 2D metric definition was changed.
- No final-claim gate semantics were broadened.

## Required validation

Before expanding to all windows, verify W045 profile-only output against original V7-z W45 behavior, especially P/V/H/Je/Jw object-window candidates and selected main candidates.
