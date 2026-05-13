# V7-y H/Jw pattern-only shape-normalized V6_1-style detector

## Purpose
Test whether H/Jw shape-normalized profile patterns contain object-level pattern windows over day0-70 while reusing the original W45 detector skeleton.

## What was changed relative to system W45
- object_scope: H-only and Jw-only runs
- detector_day_range: day0-70
- detector input: raw profile state -> shape-normalized profile pattern

## What was not changed
- profile representation: 2-degree latitudinal profiles
- yearwise/daywise shape normalization before day0-70 detection
- ruptures.Window detector parameters
- local peak, band, window merge, and bootstrap support logic

## System W45 reference
- W002: day40-day48, anchor day 45

## Object summaries
### H
- n_windows: 3
- n_accepted_95_windows: 1
- main_peak_days: [38, 48, 58]
- pre-W45 peak days: [38]
- within-W45 peak days: [48]
- post-W45 peak days: [58]

### Jw
- n_windows: 2
- n_accepted_95_windows: 0
- main_peak_days: [16, 40]
- pre-W45 peak days: [16]
- within-W45 peak days: [40]
- post-W45 peak days: []

## Interpretation rule
This run identifies shape-pattern detector peaks/windows. It does not infer causality, physical mechanism, or H/Jw pathway order.