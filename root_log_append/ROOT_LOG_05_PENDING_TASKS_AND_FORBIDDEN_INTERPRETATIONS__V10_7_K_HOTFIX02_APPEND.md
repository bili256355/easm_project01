# ROOT_LOG_05 append — V10.7_k HOTFIX02 pending checks

- HOTFIX02 is a performance patch only. It does not make previous 10/10 trial runs interpretable.
- Do not interpret V10.7_k as formal evidence unless run_meta confirms a high-count run, e.g. `n_permutation=5000`, `n_bootstrap=1000`.
- If CPU is still low after HOTFIX02, the remaining bottleneck is likely the serial multivariate ridge / remove-one-source layer; that would require a separate HOTFIX03 or a reduced formal diagnostic mode.
