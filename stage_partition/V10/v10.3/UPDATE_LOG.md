# V10.3 update log

## peak_discovery_sensitivity_v10_3

Purpose:

- Add a new V10 subfolder for joint + object-native free-discovery sensitivity tests.
- Build on the verified V10.1 joint lineage and V10.2 object-native catalogs.
- Use single-factor perturbations rather than mixed-grid interpretation.
- Keep physical interpretation and pair-order analysis out of this layer.

Boundary:

- Does not re-decide strict accepted windows.
- Does not classify physical sub-peaks.
- Does not import V6/V6_1/V7/V9 modules.
- Dynamically loads local V10.2 semantic base by path to reuse the verified free-discovery implementation.


## HOTFIX02 - 2026-05-12

- Fixed the same Python dynamic-import/dataclass registration issue inside `load_v10_2_module()` in `src/peak_discovery_sensitivity_v10_3.py`.
- HOTFIX01 registered the V10.3 runner module itself; HOTFIX02 additionally registers the dynamically loaded V10.2 semantic-base module before `exec_module()`.
- No detector, bootstrap, band, merge, lineage, or sensitivity-test semantics were changed.

## HOTFIX03 - 2026-05-12

- Suppressed pandas `FutureWarning` from object-dtype `.fillna(False).astype(bool)` in the scope/config summary helper.
- Added explicit nullable-boolean normalization via `_safe_bool_true_count()`.
- This is warning cleanup only; it does not change detector, bootstrap, candidate, band, merge, lineage, or sensitivity-test semantics.

## HOTFIX04 / smooth-input extension - 2026-05-12

- Added an explicit `smooth_input` sensitivity group to V10.3.
- New config: `SMOOTH_INPUT_5D`, comparing the baseline smooth9 input against a smooth5 `smoothed_fields.npz` input.
- The smooth5 path is resolved from `V10_3_SMOOTH5_FIELDS` first; otherwise the default project path is `foundation/V1/outputs/baseline_smooth5_a/preprocess/smoothed_fields.npz` with a few common fallback tags.
- Added `input_source` and `smoothed_fields_path` columns to per-config outputs and scope summaries.
- Added `cross_scope/input_source_inventory_v10_3.csv` so the run records exactly which input files were loaded.
- The smooth-input test remains method-layer sensitivity only: it does not add physical interpretation, pair-order analysis, or accepted-window re-decision.

## HOTFIX05 / smooth5-internal parameter sensitivity extension

- Added smooth5-internal single-factor sensitivity configs under the existing V10.3 runner.
- `SMOOTH_INPUT_5D` remains the smooth9-vs-smooth5 input comparison.
- New `SMOOTH5_*` configs compare against `SMOOTH_INPUT_5D`, not against smooth9 `BASELINE`, so they test whether the detector-width-dominated sensitivity pattern persists inside 5-day smoothed input.
- Added `reference_config_id` to comparison outputs to make the comparison baseline explicit.
- Added `cross_scope/smooth5_internal_sensitivity_summary_by_scope_factor_v10_3.csv`.
- No physical interpretation, pair-order analysis, or accepted-window re-decision is introduced.

## HOTFIX06 closure audits

- Added formal candidate order inversion audit: `cross_scope/candidate_order_inversion_summary_v10_3.csv`.
- Added detector-width minimal bootstrap closure in the default `baseline_and_match` mode for `DET_WIDTH_16`, `DET_WIDTH_24`, `SMOOTH5_DET_WIDTH_16`, and `SMOOTH5_DET_WIDTH_24`.
- Added detector-width bootstrap support summary: `cross_scope/detector_width_bootstrap_support_summary_v10_3.csv`.
- Added explicit candidate shift-type summary to separate same-day, <=2d, <=5d, <=8d, missing, and new-candidate changes: `cross_scope/candidate_shift_type_summary_v10_3.csv`.
- Added new perturbed candidate inventory with lineage/provenance fields: `lineage_mapping/new_candidate_inventory_v10_3.csv`.
- No detector, bootstrap, candidate-band, derived-window, or lineage algorithm semantics were changed except that the default bootstrap mode now includes the minimal detector-width bootstrap closure.
