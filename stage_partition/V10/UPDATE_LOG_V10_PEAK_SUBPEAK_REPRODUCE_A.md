# UPDATE LOG: V10 peak_subpeak_reproduce_v10_a

## Purpose

Create an independent semantic rewrite of the V9 subpeak/peak extraction layer.

## User decision honored

All previous W045 Jw/H sensitivity-cause audit judgments are treated as discarded and are not inherited.

## Added

- `V10/scripts/run_peak_subpeak_reproduce_v10_a.py`
- `V10/src/stage_partition_v10/__init__.py`
- `V10/src/stage_partition_v10/peak_subpeak_reproduce_v10_a.py`
- `V10/README_V10_PEAK_SUBPEAK_REPRODUCE_A.md`

## Engineering boundary

The implementation does not import or call V9/V7 modules. It only reads V9 output CSVs for regression comparison.

## Output

- `V10/outputs/peak_subpeak_reproduce_v10_a`
- `V10/logs/peak_subpeak_reproduce_v10_a/last_run.txt`

## Interpretation boundary

This patch is not a scientific interpretation patch. Its first task is to test exact reproduction against V9 outputs.
