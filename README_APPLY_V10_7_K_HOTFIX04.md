# V10.7_k HOTFIX04 — avoid stage-5 multivariate ridge bottleneck

This hotfix keeps HOTFIX01–03 behavior and adds runtime control for the expensive stage 5/6 ridge layers.

New CLI arguments:

```bat
--multivariate-policy full|fast|skip
--multivariate-n-perm N
--object-contribution-policy full|fast|skip
```

Recommended H-focused run when stage 5 is too slow:

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_structure_transition_mapping_v10_7_k.py --n-perm 5000 --n-boot 1000 --n-jobs 8 --progress-every 20 --pairwise-scope h-source --pairwise-bootstrap-policy candidate --multivariate-policy skip --object-contribution-policy skip
```

This run evaluates H_E2 structure metrics against all M targets through pairwise same-year vs shuffled-year tests. It intentionally does not evaluate overall E2→M multivariate ridge skill or remove-one-source contribution.

If a light multivariate diagnostic is needed:

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_structure_transition_mapping_v10_7_k.py --n-perm 5000 --n-boot 1000 --n-jobs 8 --progress-every 20 --pairwise-scope h-source --pairwise-bootstrap-policy candidate --multivariate-policy fast --multivariate-n-perm 200 --object-contribution-policy skip
```

Interpretation boundary:
- `multivariate-policy skip` means do not use mapping skill / object contribution tables for route decisions.
- Use pairwise and H-specific mapping tables only.
