# UPDATE LOG — V8 state_relation_v8_a

## Purpose

Adds a first isolated **state-only** relation layer on top of the clean V8 peak-only baseline.

This patch is intentionally not a growth/process patch. It excludes:

- growth curves and ΔG diagnostics;
- rollback / multi-stage / alternating-growth logic;
- process_a labels;
- final causal or process claims.

## Method boundary

The state layer uses the original V7 pre-post state definitions for `S_dist` and `S_pattern`. The new object of diagnosis is pairwise:

```text
ΔS_AB(t) = S_A(t) - S_B(t)
```

The first implementation treats state relation as a segment problem:

```text
raw atomic segment → classified segment → relation block
```

## Key methodological choices

1. **No fixed |ΔS| strength threshold** is used for near or dominant classification.
2. Segment strength is compared to a **same-object same-size bootstrap reproducibility null**.
3. Dominant classification uses segment signed mean / absolute mean separation.
4. Near segment classification is first established at segment level; near blocks are aggregated from near segments.
5. Uncertain segments can aggregate into uncertain blocks.
6. Bootstrap segment matching is overlap-based and allows boundary drift / split-match rather than exact day-wise equality.

## Important outputs

```text
V8/outputs/state_relation_v8_a/per_window/W045/object_state_curves_W045.csv
V8/outputs/state_relation_v8_a/per_window/W045/pairwise_delta_state_curves_W045.csv
V8/outputs/state_relation_v8_a/per_window/W045/pairwise_state_raw_segments_observed_W045.csv
V8/outputs/state_relation_v8_a/per_window/W045/pairwise_state_segments_observed_W045.csv
V8/outputs/state_relation_v8_a/per_window/W045/state_reproducibility_null_segment_summary_W045.csv
V8/outputs/state_relation_v8_a/per_window/W045/pairwise_state_relation_blocks_W045.csv
V8/outputs/state_relation_v8_a/per_window/W045/pairwise_state_segment_bootstrap_W045.csv
V8/outputs/state_relation_v8_a/per_window/W045/pairwise_state_block_bootstrap_W045.csv
V8/outputs/state_relation_v8_a/per_window/W045/multi_object_state_relation_network_W045.csv
V8/outputs/state_relation_v8_a/per_window/W045/peak_state_relation_comparison_W045.csv
```

## Risk notes

This is a method-test patch. The outputs should be audited for:

- whether same-object reproducibility null is too wide or too narrow;
- whether segment construction parameters fragment or over-merge the curves;
- whether bootstrap matching is too lenient or too strict;
- whether near segments are correctly prevented from being formed by cancellation of large positive/negative segments.

## Hotfix01 — boundary NaN awareness

### Reason

Smoothed fields have legitimate leading/trailing NaN days at the season boundaries. W045 can be affected at the beginning of the analysis/pre-reference range, and the last accepted window can be affected near season end. The previous state_relation_v8_a patch only skipped NaN days during segmentation and did not audit them explicitly, which allowed repeated all-NaN mean warnings and made the valid analysis domain implicit.

### Changes

- Added warning-safe `nanmean` handling for expected all-NaN boundary slices.
- Added explicit profile/reference validity audit:

```text
state_profile_reference_validity_audit_<window>.csv
```

- Added explicit state-curve finite-day audit:

```text
state_valid_day_audit_<window>.csv
```

- Added explicit pairwise ΔS common-valid-domain audit:

```text
pairwise_delta_state_valid_domain_audit_<window>.csv
```

- Segment extraction remains finite-domain based: boundary NaNs are audited and skipped, not interpreted as relation gaps or relation transitions.
- The method still excludes growth, rollback, multi-stage, process_a, and causal/process claims.

### Interpretation note

If the audits show leading/trailing NaN days inside a nominal pre/post/reference range, then the effective finite reference days are shorter than the nominal range. This does not automatically invalidate the run, but the affected window/baseline/branch must be treated as boundary-affected when interpreting state segments.
