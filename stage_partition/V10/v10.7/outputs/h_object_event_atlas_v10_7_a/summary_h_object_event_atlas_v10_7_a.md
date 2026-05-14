# V10.7_a H-only main-method event atlas summary

## Scope

V10.7_a is a H-only main-method baseline export. It reruns the object-native detector semantics for H across multiple detector_width values and exports full-season score curves, candidate catalogs, width-stability tables, and strong-window H event packages.

It is a method-layer baseline for later tests. It is not a physical interpretation layer, not a yearwise validation, not a spatial validation, and not a causal test.

## Baseline reproduction check

Expected width=20 H candidates: [19, 35, 57, 77, 95, 115, 129, 155]
All expected candidates matched within ±2 days: **True**

## Strong-window H role overview

### W045
- width=12: pre_and_inside_candidate | pre=36; inside=46; post=54; nearest=46 (distance=1)
- width=16: pre_and_post_without_inside | pre=36; inside=none; post=55; nearest=36 (distance=9)
- width=20: pre_and_post_without_inside | pre=35; inside=none; post=57; nearest=35 (distance=10)
- width=24: pre_window_candidate_only | pre=34; inside=none; post=none; nearest=34 (distance=11)
- width=28: window_absent | pre=none; inside=none; post=none; nearest=77 (distance=32)

### W081
- width=12: inside_and_post_candidate | pre=none; inside=77; post=95; nearest=77 (distance=4)
- width=16: pre_and_inside_candidate | pre=55; inside=77; post=95; nearest=77 (distance=4)
- width=20: pre_and_inside_candidate | pre=57; inside=77; post=95; nearest=77 (distance=4)
- width=24: inside_and_post_candidate | pre=none; inside=77; post=95; nearest=77 (distance=4)
- width=28: inside_and_post_candidate | pre=none; inside=77; post=96; nearest=77 (distance=4)

### W113
- width=12: pre_and_inside_candidate | pre=95; inside=114; post=129; nearest=114 (distance=1)
- width=16: pre_and_inside_candidate | pre=95; inside=115; post=129; nearest=115 (distance=2)
- width=20: pre_and_inside_candidate | pre=95; inside=115; post=129; nearest=115 (distance=2)
- width=24: pre_and_inside_candidate | pre=95; inside=115; post=131; nearest=115 (distance=2)
- width=28: pre_and_inside_candidate | pre=96; inside=114; post=132; nearest=114 (distance=1)

### W160
- width=12: pre_and_inside_candidate | pre=142; inside=155;164; post=171; nearest=164 (distance=4)
- width=16: pre_and_inside_candidate | pre=143; inside=155; post=none; nearest=155 (distance=5)
- width=20: inside_window_main_candidate | pre=none; inside=155; post=none; nearest=155 (distance=5)
- width=24: pre_window_candidate_only | pre=154; inside=none; post=none; nearest=154 (distance=6)
- width=28: pre_window_candidate_only | pre=153; inside=none; post=none; nearest=153 (distance=7)

## Width-stability note

Width-stable baseline candidates: 7 / 8

- baseline day 19: matched_widths=12;16;20;24; max_shift_abs=0.0; stable=True
- baseline day 35: matched_widths=12;16;20;24; max_shift_abs=1.0; stable=True
- baseline day 57: matched_widths=16;20; max_shift_abs=2.0; stable=False
- baseline day 77: matched_widths=12;16;20;24;28; max_shift_abs=0.0; stable=True
- baseline day 95: matched_widths=12;16;20;24;28; max_shift_abs=1.0; stable=True
- baseline day 115: matched_widths=12;16;20;24;28; max_shift_abs=1.0; stable=True
- baseline day 129: matched_widths=12;16;20;24; max_shift_abs=2.0; stable=True
- baseline day 155: matched_widths=12;16;20;24;28; max_shift_abs=2.0; stable=True

## Interpretation boundary

Do not use V10.7_a alone to claim H weak precursor, H condition, or physical mechanism. Use it to choose target H event packages for later yearwise, spatial, or conditional tests.
