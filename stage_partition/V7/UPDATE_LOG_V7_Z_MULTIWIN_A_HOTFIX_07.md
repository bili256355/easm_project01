# V7-z-multiwin-a hotfix_07 — bootstrap index NameError fix

## Purpose
Fix a runtime failure introduced in hotfix_06 W45 profile order tests.

## Error
`_bootstrap_pairwise_metrics()` referenced `boot_indices` without defining it:

```text
NameError: name 'boot_indices' is not defined
```

## Fix
Inside `_bootstrap_pairwise_metrics()`:

- Define `ny` from the profile year dimension.
- Build `boot_indices = _make_bootstrap_indices(ny, scope, cfg)` before the bootstrap loop.
- Keep paired year-resampling semantics across objects.

## Scientific scope
No scientific definitions changed.

Unchanged:
- W45 default mode
- profile-only default
- 2D disabled by default
- detector semantics: z-scored climatological profile + ruptures.Window
- C0/C1/C2 baselines
- S_dist / S_pattern / V_dist / V_pattern definitions
- hotfix_06 order-test logic
