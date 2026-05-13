# V10.2 object-native peak discovery

This subpackage runs object-native full-season peak discovery for P/V/H/Je/Jw using the same free-season discovery semantics that were verified in V10.1 for the joint-object main-window chain.

It is deliberately **not** a sensitivity test and **not** a physical interpretation layer.

## Scope

- Build object profiles for P, V, H, Je, Jw from `smoothed_fields.npz`.
- For each object independently:
  - build a full-season object-native state matrix;
  - run `ruptures.Window(width=20, model="l2", min_size=2, jump=1, pen=4.0)`;
  - extract detector local peak candidates;
  - run bootstrap recurrence support;
  - build candidate bands and object-derived windows;
  - map candidates to V10.1 joint lineage and V10 window-conditioned object peaks.

## Out of scope

- No accepted-window sensitivity test.
- No pair-order analysis.
- No physical subpeak classification.
- No redecision of strict accepted windows.

## Run

```bat
python D:\easm_project01\stage_partition\V10\v10.2\scripts\run_object_native_peak_discovery_v10_2.py
```

Debug with fewer bootstrap samples:

```bat
set V10_2_DEBUG_N_BOOTSTRAP=20
python D:\easm_project01\stage_partition\V10\v10.2\scripts\run_object_native_peak_discovery_v10_2.py
```

Formal run:

```bat
set V10_2_N_BOOTSTRAP=1000
python D:\easm_project01\stage_partition\V10\v10.2\scripts\run_object_native_peak_discovery_v10_2.py
```

## Main outputs

All outputs are contained under:

```text
V10\v10.2\outputs\object_native_peak_discovery_v10_2
```

Key files:

```text
cross_object/object_native_candidate_catalog_all_objects_v10_2.csv
cross_object/object_native_derived_windows_all_objects_v10_2.csv
lineage_mapping/object_candidate_to_joint_lineage_v10_2.csv
cross_object/object_native_peak_summary_v10_2.csv
OBJECT_NATIVE_PEAK_DISCOVERY_V10_2_SUMMARY.md
```
