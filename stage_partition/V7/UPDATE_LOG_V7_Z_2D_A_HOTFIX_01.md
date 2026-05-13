# UPDATE LOG — V7-z-2d-a hotfix_01 performance vectorization

## Purpose
Fix low CPU utilization / slow runtime in `W45_2d_prepost_metric_mirror_v7_z_2d_a`.

## Root cause
The original 2D mirror implementation was scientifically consistent but performance-poor because the bootstrap loop repeatedly:

- rebuilt pandas DataFrames for every bootstrap sample;
- recomputed 2D distance/correlation through scalar Python functions;
- looped over object × baseline × day with high Python overhead.

This made runtime dominated by Python interpreter overhead, so CPU utilization could remain low.

## Fix
This hotfix keeps the scientific definitions unchanged, but rewrites the heavy bootstrap path to use vectorized NumPy operations:

- flatten each object-specific 2D region once;
- compute per-day weighted 2D distance/correlation as matrix operations;
- summarize bootstrap metrics directly from arrays without constructing bootstrap DataFrames;
- add visible bootstrap progress logs.

## Unchanged
- No 2D change-point detection is added.
- No clean-mainline final claims are rewritten.
- Definitions of `S_dist_2d`, `R_diff_2d`, `S_pattern_2d`, `V_dist_2d`, and `V_pattern_2d` remain unchanged.
- Output filenames and schema are preserved.

## Expected effect
Substantially lower Python overhead and better CPU utilization during the paired year-bootstrap stage.
