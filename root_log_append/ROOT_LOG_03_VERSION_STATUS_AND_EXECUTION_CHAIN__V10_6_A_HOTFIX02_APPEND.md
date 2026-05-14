# ROOT_LOG_03 append: V10.6_a HOTFIX02

## V10.6_a HOTFIX02 status

- Version: V10.6_a HOTFIX02
- Type: interpretation-classification hotfix
- Entry script unchanged:
  `stage_partition/V10/v10.6/scripts/run_w045_precluster_audit_v10_6_a.py`
- Output directory unchanged:
  `stage_partition/V10/v10.6/outputs/w045_precluster_audit_v10_6_a`
- Files replaced:
  - `curve_metrics.py`
  - `role_classifier.py`
  - `summary_writer.py`
  - `plotting.py`
- HOTFIX01 NumPy trapezoid compatibility is included cumulatively in this hotfix.

## Reason

The original V10.6_a output grouped `candidate_inside_cluster` and `curve_peak_without_marker` together as `active_or_curve objects`. This was too broad and could overstate E2 participation by `joint_all` and `Jw`.

HOTFIX02 separates:

- marker-supported active evidence;
- curve-only/ramp-or-shoulder evidence;
- weak/absent/missing evidence.

## Expected interpretation after rerun

- E1 marker-supported core objects: `joint_all/P/V/H`.
- E2 marker-supported core objects: `P/V/H/Je`.
- E2 curve-only/ramp objects: `joint_all/Jw`.
- M marker-supported core objects: `joint_all/P/V/Je/Jw`.
- H is absent from the M marker-supported core and remains a non-confirmed weak precursor.
