# V10.5_d strength-stability audit summary

## Scope

This audit separates peak strength from year-bootstrap stability.
It asks whether non-strict main-method peaks are high-score but lower-bootstrap, and whether profile-energy top1 families are stable under year resampling.
It does not reinterpret physical mechanisms or redefine accepted windows.

## Run settings

- profile_energy_bootstrap_n: `1000`
- profile_energy_bootstrap_seed: `42`
- energy_k values: `[7, 9, 11]`

## Main-method strength-stability classes

```json
{
  "LOWER_SCORE_LOWER_BOOTSTRAP": 4,
  "HIGH_SCORE_HIGH_BOOTSTRAP": 3,
  "LOWER_SCORE_HIGH_BOOTSTRAP": 1
}
```

## Profile-energy family bootstrap statuses

```json
{
  "STABLE_TOPK_FAMILY": 75,
  "STABLE_TOP1_FAMILY": 10,
  "WEAK_TOPK_FAMILY": 3,
  "UNSTABLE_ENERGY_FAMILY": 2
}
```

## Pair comparison interpretation classes

```json
{
  "BOTH_STABLE_MULTIFAMILY_STRUCTURE": 40,
  "TOP1_STRONG_AND_STABLE": 3,
  "ASSIGNED_SECONDARY_BUT_STABLE": 2
}
```

## Key output files

- `profile_validation/main_method_peak_strength_vs_bootstrap_v10_5_d.csv`
- `profile_validation/profile_energy_family_bootstrap_stability_v10_5_d.csv`
- `profile_validation/profile_energy_family_pair_comparison_v10_5_d.csv`
- `profile_validation/candidate_family_strength_stability_matrix_v10_5_d.csv`
- `profile_validation/key_competition_cases_v10_5_d.csv`

## Interpretation boundary

A high profile-energy score is not equivalent to strict accepted status. A high detector score is not equivalent to bootstrap stability. These tables are a method-layer audit only.