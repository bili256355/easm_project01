# V10.6_a HOTFIX02: participation tiers and summary wording

## Purpose

This hotfix fixes an interpretation-classification problem in V10.6_a outputs.

The original V10.6_a summary grouped `candidate_inside_cluster` and `curve_peak_without_marker` together as `active_or_curve objects`. This was too broad for W045 interpretation: it could make E2 look as if `joint_all` and `Jw` were active in the same sense as P/V/H/Je, even though they only showed curve-only/ramp-like evidence without an in-cluster candidate marker.

HOTFIX02 separates event-semantics evidence into three tiers:

1. `marker_supported_active`: has an in-cluster candidate marker; use as event-semantics core.
2. `curve_only_ramp_or_shoulder`: curve rises or peaks in the fixed cluster but no candidate marker; use as ramp/shoulder/background evidence only.
3. `weak_curve_signal` / `absent_or_missing`: weak, absent, or missing evidence.

## Files to replace

Copy these files into the existing V10.6_a package, replacing the old files:

```text
stage_partition/V10/v10.6/src/stage_partition_v10_6/curve_metrics.py
stage_partition/V10/v10.6/src/stage_partition_v10_6/role_classifier.py
stage_partition/V10/v10.6/src/stage_partition_v10_6/summary_writer.py
stage_partition/V10/v10.6/src/stage_partition_v10_6/plotting.py
```

This hotfix includes the HOTFIX01 NumPy trapezoid compatibility fix inside `curve_metrics.py`; you do not need to reapply HOTFIX01 separately after applying HOTFIX02.

## What changes in outputs

After rerunning the same entry script, these outputs will be regenerated with safer wording/columns:

```text
summary_w045_precluster_audit_v10_6_a.md
tables/w045_object_cluster_metrics_v10_6_a.csv
tables/w045_cluster_participation_matrix_v10_6_a.csv
tables/w045_interpretation_summary_v10_6_a.csv
figures/w045_cluster_participation_heatmap_v10_6_a.png
```

The key corrected interpretation should be:

```text
E1 marker-supported core objects: joint_all/P/V/H
E2 marker-supported core objects: P/V/H/Je
E2 curve-only/ramp objects: joint_all/Jw
M marker-supported core objects: joint_all/P/V/Je/Jw
M absent/missing: H
```

## Run command after replacement

```bash
python D:\easm_project01\stage_partition\V10\v10.6\scripts\run_w045_precluster_audit_v10_6_a.py
```

## Interpretation boundary

HOTFIX02 does not alter the core scientific design or raw metrics. It only prevents curve-only evidence from being interpreted as equal to marker-supported event participation. H day35 remains not supported as a confirmed weak precursor by V10.6_a alone.
