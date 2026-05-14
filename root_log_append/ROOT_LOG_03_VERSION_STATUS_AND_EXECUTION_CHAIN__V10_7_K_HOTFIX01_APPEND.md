# ROOT_LOG_03 append — V10.7_k HOTFIX01 parameter override

- Patch type: runtime parameter hotfix for V10.7_k.
- Reason: environment-variable commands did not reliably override `n_permutation` and `n_bootstrap`; run_meta remained at 10/10.
- Change: entry script now accepts explicit CLI arguments `--n-perm` and `--n-boot`, and passes them directly into the pipeline.
- Scientific semantics: unchanged. Same output directory and same analysis logic; only runtime resampling counts are made controllable.
- Verification requirement: after running, inspect `run_meta/run_meta_v10_7_k.json` and confirm `n_permutation` and `n_bootstrap` match the requested values.
