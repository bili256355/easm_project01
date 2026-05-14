# ROOT_LOG_05 append — V10.7_k HOTFIX01 pending checks

- Do not interpret V10.7_k output as formal evidence unless run_meta confirms a high-count run, e.g. `n_permutation=5000` and `n_bootstrap=1000`.
- If run_meta still reports 10/10 after HOTFIX01, the replacement files were not applied to the executed engineering tree or a different entry script was used.
- The previous 10/10 runs remain trial/debug runs only.
