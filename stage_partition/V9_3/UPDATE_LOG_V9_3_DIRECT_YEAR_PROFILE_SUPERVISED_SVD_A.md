# UPDATE_LOG_V9_3_DIRECT_YEAR_PROFILE_SUPERVISED_SVD_A

## Version

`v9_3_a = direct-year profile-evolution supervised SVD / PLS1`

## Why this branch exists

V9.2 used direct-year 2D spatiotemporal MVEOF and then checked whether PC high/low year groups had different group-composite peak timing. That avoids yearly peak as Y, but it is unsupervised and relatively expensive.

V9.3_a opens a different control line requested by the user:

1. Do not use bootstrap year-resampling as samples.
2. Do not use 2D-field X, to reduce runtime.
3. Use true yearly peak timing / peak-order contrast as Y.
4. Use supervised SVD / PLS1 to find profile-evolution modes associated with yearly peak-order variation.
5. Treat result usability assessment as a primary output, not a post-hoc add-on.

## Core implementation

- Sample axis: real years 1979-2023.
- X: five-object yearly profile-evolution anomaly matrix, `year x (object, day, profile_coord)`.
- Y: yearly pair peak delta, `peak_B(year) - peak_A(year)`.
- Method: one-target supervised SVD / PLS1, `u = normalize(X^T Y)` and `score = X u`.
- Target sets: priority V9.1_f targets plus full 10 object pairs per window by default.
- Built-in audits:
  - yearly object peak registry;
  - yearly pair peak delta registry;
  - Y quality audit;
  - permutation audit;
  - leave-one-year influence audit;
  - split-half stability audit;
  - score high/low phase separation audit;
  - result usability registry.

## Interpretation boundaries

- V9.3_a is target-guided; it is not an unsupervised natural mode.
- Yearly peak noise is a core risk; use `v9_3_result_usability_registry.csv` first.
- `dominant_object` is a loading summary, not a physical driver.
- Score high/low years are statistical score phases, not named physical year types.
- This branch does not produce physical mechanism conclusions.

## Default run command

```bat
python D:\easm_project01\stage_partition\V9_3\scripts\run_direct_year_profile_supervised_svd_v9_3_a.py
```

## Optional fast-run environment settings

```bat
set V9_3_TARGET_MODE=priority_only
set V9_3_PERM_N=300
set V9_3_SPLIT_HALF_N=50
python D:\easm_project01\stage_partition\V9_3\scripts\run_direct_year_profile_supervised_svd_v9_3_a.py
```

## Detector compatibility note

By default the V7 detector is called with `bootstrap_n=0`; V9.3_a does not use bootstrap as evidence. If a local V7 helper requires at least one bootstrap only for API compatibility, enable:

```bat
set V9_3_ALLOW_DETECTOR_COMPAT_BOOTSTRAP_RETRY=1
set V9_3_COMPAT_DETECTOR_BOOTSTRAP_N=1
```

The compatibility bootstrap is ignored by V9.3 outputs.
