# V10.7_k HOTFIX02 — pairwise permutation/bootstrap CPU speedup

## Why this hotfix exists
The formal V10.7_k run with `--n-perm 5000 --n-boot 1000` can be slow and may use little CPU because the pairwise mapping layer originally evaluated thousands of source-target pairs serially in one Python process.

## What changed
- Keeps HOTFIX01 explicit CLI overrides:
  - `--n-perm`
  - `--n-boot`
- Adds:
  - `--n-jobs`
- The pairwise source-target permutation/bootstrap tests are distributed across worker processes when `--n-jobs > 1`.
- Scientific semantics of the pairwise tests are unchanged: each pair still uses the same Spearman permutation test and bootstrap CI logic; only independent pairs are run in parallel.
- Multivariate ridge mapping and remove-one-source contribution remain unchanged in this conservative hotfix.

## Apply
Unzip this patch at the project root:

```text
D:\easm_project01
```

It replaces:

```text
stage_partition\V10\v10.7\scripts\run_w45_structure_transition_mapping_v10_7_k.py
stage_partition\V10\v10.7\src\stage_partition_v10_7\w45_structure_transition_pipeline.py
```

## Quick verification
First verify parameter passing and worker startup with small counts:

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_structure_transition_mapping_v10_7_k.py --n-perm 37 --n-boot 23 --n-jobs 4
```

Check:

```text
D:\easm_project01\stage_partition\V10\v10.7\outputs\w45_structure_transition_mapping_v10_7_k\run_meta\run_meta_v10_7_k.json
```

It should show:

```json
"n_permutation": 37,
"n_bootstrap": 23,
"n_jobs": 4
```

## Formal run suggestion
Use a conservative worker count first. On a 16-core CPU, start with 8 workers:

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_structure_transition_mapping_v10_7_k.py --n-perm 5000 --n-boot 1000 --n-jobs 8
```

If the machine remains responsive and memory is fine, try increasing `--n-jobs`.
