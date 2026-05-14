# V10.7_k HOTFIX03: progress + runtime controls

Purpose:
- Add visible stage progress and pairwise progress prints.
- Preserve HOTFIX01 explicit `--n-perm/--n-boot` overrides.
- Preserve HOTFIX02 `--n-jobs` pairwise parallelism.
- Add `--progress-every`, `--pairwise-scope`, and `--pairwise-bootstrap-policy`.

Apply by copying this patch into the project root, preserving paths.

## First validation run

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_structure_transition_mapping_v10_7_k.py --n-perm 37 --n-boot 23 --n-jobs 4 --progress-every 10 --pairwise-scope h-source --pairwise-bootstrap-policy candidate
```

Confirm run_meta has:
- n_permutation = 37
- n_bootstrap = 23
- n_jobs = 4
- progress_every = 10
- pairwise_scope = h-source
- pairwise_bootstrap_policy = candidate

## Recommended H-focused formal run

This is the practical first formal run for the current H-specific question. It computes pairwise tests only for H as the E2 source, while keeping multivariate/object contribution layers.

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_structure_transition_mapping_v10_7_k.py --n-perm 5000 --n-boot 1000 --n-jobs 8 --progress-every 20 --pairwise-scope h-source --pairwise-bootstrap-policy candidate
```

## Full all-pairs formal run

This can be much slower.

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_structure_transition_mapping_v10_7_k.py --n-perm 5000 --n-boot 1000 --n-jobs 8 --progress-every 50 --pairwise-scope all --pairwise-bootstrap-policy candidate
```

Use `--pairwise-bootstrap-policy all` only if full bootstrap CI for every pair is required; it is much slower.

## Notes

- `candidate` bootstrap policy computes bootstrap CI only for weak/clear pairwise candidates; non-candidate pairwise rows get NaN CI. Permutation p is still computed for all included pairs.
- `h-source` limits pairwise rows to H_E2 source metrics mapping to all M targets. This is appropriate for the current H-specific structural question.
- This patch does not alter scientific definitions of structure metrics or ridge mapping.
