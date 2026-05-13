# stage_partition/V7

## Version role

`V7` is an independent diagnostic branch built on top of the accepted stage-partition time-object layer.

Its first task is deliberately narrow:

> For the four accepted transition windows, diagnose when each field-level state reaches its strongest detector-profile transition signal.

The accepted windows are inherited from V6/V6_1 logs and outputs:

- 45 window
- 81 window
- 113 window
- 160 window

Candidate/sub-threshold peaks such as 18, 96, 132, and 135 are not used by this V7-a diagnostic.

## What V7-a does

`field_transition_timing_v7_a`:

1. reads V6/V6_1 logs and output tables as audit evidence;
2. verifies the four accepted bootstrap-supported points;
3. reads the V6_1 derived window table and keeps only windows anchored at 45, 81, 113, and 160;
4. rebuilds the original stage-partition profiles from `smoothed_fields.npz`;
5. constructs field-only state matrices for P, V, H, Je, and Jw;
6. applies the same `ruptures.Window` score-profile semantics as V6;
7. extracts each field's peak score day inside each accepted transition window;
8. writes tables, audit logs, and optional score-profile plots.

## What V7-a does not do

V7-a does not:

- re-decide which transition windows are accepted;
- include all candidate windows;
- use downstream lead-lag or pathway results;
- infer causal direction;
- diagnose spatial earliest/late regions;
- test whether timing order is consistent across windows as a scientific claim.

## Main runner

```bat
python D:\easm_project01\stage_partition\V7\scripts\run_field_transition_timing_v7_a.py
```

## Main output

```text
D:\easm_project01\stage_partition\V7\outputs\field_transition_timing_v7_a\field_transition_peak_days_by_window.csv
```

This table is the primary output for the first question.
