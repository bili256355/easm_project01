
# V10.7_k HOTFIX01 — explicit n_perm / n_boot CLI override

## Why this hotfix exists
The previous run instructions relied on environment variables:

```bat
set V10_7_K_N_PERM=5000
set V10_7_K_N_BOOT=1000
python ...run_w45_structure_transition_mapping_v10_7_k.py
```

The user's run_meta still reported 10/10, so the runtime override was not reliably applied.

## What changed
- `scripts/run_w45_structure_transition_mapping_v10_7_k.py` now accepts:
  - `--n-perm`
  - `--n-boot`
- The entry script passes these values directly into the pipeline.
- The pipeline function accepts direct overrides and writes the actual values into run_meta.

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

## Recommended formal run

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_structure_transition_mapping_v10_7_k.py --n-perm 5000 --n-boot 1000
```

## Quick verification run
Use a small unusual number first:

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_structure_transition_mapping_v10_7_k.py --n-perm 37 --n-boot 23
```

Then inspect:

```text
D:\easm_project01\stage_partition\V10\v10.7\outputs\w45_structure_transition_mapping_v10_7_k\run_meta\run_meta_v10_7_k.json
```

It should show:

```json
"n_permutation": 37,
"n_bootstrap": 23
```

If it does, run the formal 5000/1000 command.
