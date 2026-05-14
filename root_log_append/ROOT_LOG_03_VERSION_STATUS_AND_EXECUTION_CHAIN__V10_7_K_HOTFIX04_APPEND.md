# ROOT LOG APPEND — V10.7_k HOTFIX04

V10.7_k HOTFIX04 adds runtime controls for the stage-5 multivariate ridge bottleneck. It introduces `--multivariate-policy full|fast|skip`, `--multivariate-n-perm`, and `--object-contribution-policy full|fast|skip` while preserving HOTFIX01 explicit parameter overrides and HOTFIX03 progress / pairwise scope controls.

Reason: user observed the run stalls for a long time at `stage 5/9 multivariate ridge mapping`. The immediate research priority is to confirm H-specific structural pairwise lines, so the full multivariate ridge layer should be optional rather than blocking the whole run.
