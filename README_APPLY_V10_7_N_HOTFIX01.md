# V10.7_n HOTFIX01 performance patch

## Purpose

This hotfix addresses slow execution and low CPU usage during:

```text
[V10.7_n] experiment A: local-vs-background relations
```

The original V10.7_n performed many source-target evaluations with Python-level permutation/bootstrap loops, and experiment A also ran expensive incremental-CV and sliding-window tables under the full formal `--n-perm` / `--n-boot` settings. That made the run slow while using only one Python process.

## What changed

1. Vectorized pairwise permutation correlation p-values.
2. Vectorized high/low bootstrap confidence intervals.
3. Replaced refit-for-each-left-out-year LOOCV with the OLS hat-matrix identity.
4. Added `--experiment-a-policy` to control the heaviest experiment-A branches.

## Policies

```text
--experiment-a-policy full
```

Default. Preserves the original formal resampling semantics, but with vectorized pairwise p/CI and faster LOOCV.

```text
--experiment-a-policy screen
```

Uses `--screen-n-perm` / `--screen-n-boot` for experiment-A screening tables, and keeps incremental-CV delta rankings without formal delta resampling. Use this to rapidly identify whether local E2 or broad background is promising.

```text
--experiment-a-policy skip_heavy
```

Runs the compact local-vs-background pairwise table and skips the expensive incremental-CV and sliding-window screens. Use this to check the rest of the pipeline quickly.

## Recommended commands

Formal run preserving original semantics:

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_h_zonal_width_background_target_preinfo_v10_7_n.py --n-perm 5000 --n-boot 1000 --n-random-windows 1000 --group-frac 0.30 --experiment-a-policy full --progress
```

Fast screen run:

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_h_zonal_width_background_target_preinfo_v10_7_n.py --n-perm 5000 --n-boot 1000 --n-random-windows 1000 --group-frac 0.30 --experiment-a-policy screen --screen-n-perm 199 --screen-n-boot 199 --progress
```

Fast pipeline connectivity run:

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_h_zonal_width_background_target_preinfo_v10_7_n.py --n-perm 37 --n-boot 23 --n-random-windows 50 --group-frac 0.30 --experiment-a-policy skip_heavy --progress
```

## Interpretation boundary

`screen` and `skip_heavy` are performance policies. They are not formal statistical replacements for the full run. Tables produced under these policies include policy columns where relevant.
