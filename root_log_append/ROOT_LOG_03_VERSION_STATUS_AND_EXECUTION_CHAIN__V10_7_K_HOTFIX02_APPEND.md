# ROOT_LOG_03 append — V10.7_k HOTFIX02 pairwise speedup

- Patch type: performance hotfix for V10.7_k.
- Reason: formal high-count permutation/bootstrap runs were slow and used little CPU because pairwise source-target tests ran serially.
- Change: entry script now accepts `--n-jobs`; pairwise permutation/bootstrap tests are distributed across worker processes when `n_jobs > 1`.
- Scientific semantics: unchanged for pairwise tests; this hotfix only parallelizes independent source-target pair calculations. Multivariate ridge mapping and remove-one-source contribution are not changed in this conservative patch.
- Verification requirement: after running, inspect `run_meta/run_meta_v10_7_k.json` and confirm `n_permutation`, `n_bootstrap`, and `n_jobs` match the requested values.
