# V7-w H/Jw object-specific V6_1-style window detection

## Purpose
Reuse the original W45 detection skeleton while changing only object scope and detector time range.

## What was changed relative to system W45
- object_scope: all-object system matrix -> H-only and Jw-only runs
- detector_day_range: day0-70

## What was not changed
- profile representation: 2-degree latitudinal profiles
- full-season z-score standardization before day0-70 detection
- ruptures.Window detector parameters
- local peak, band, window merge, and bootstrap support logic

## System W45 reference
- W002: day40-day48, anchor day 45

## Object summaries
### H
- n_windows: 3
- n_accepted_95_windows: 0
- main_peak_days: [19, 35, 57]
- pre-W45 peak days: [19, 35]
- within-W45 peak days: []
- post-W45 peak days: [57]

### Jw
- n_windows: 1
- n_accepted_95_windows: 0
- main_peak_days: [41]
- pre-W45 peak days: []
- within-W45 peak days: [41]
- post-W45 peak days: []

## Interpretation rule
This run identifies object-level detector peaks/windows. It does not infer causality, physical mechanism, or H/Jw pathway order.