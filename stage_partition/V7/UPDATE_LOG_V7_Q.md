# UPDATE_LOG_V7_Q

## w45_feature_process_resolution_v7_q

Purpose: raise W45 process analysis from one whole-field progress curve per object to a field × feature/component process ensemble.

Core constraints:
- Rebuilds from the current 2° interpolated profile/state base.
- Does not read V7-m, V7-n, V7-o, or V7-p derived result tables as input.
- Includes all five W45 objects: P, V, H, Je, Jw.
- Keeps weak/noisy/low-contribution features in the output and labels them instead of deleting them.
- Does not call t25 “onset”; it is retained only as `t25` / `early_progress_day_25`.
- Produces feature process markers, field feature timing distributions, pair feature-distribution relations, H/Jw detail, early-group organization, and Je feature consistency checks.

Run:

```bat
python D:\easm_project01\stage_partition\V7\scripts\run_w45_feature_process_resolution_v7_q.py
```
