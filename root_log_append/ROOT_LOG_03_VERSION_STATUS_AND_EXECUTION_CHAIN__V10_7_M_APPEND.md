# ROOT_LOG_03 append — V10.7_m execution chain

## Version

`v10.7_m = window_anchor_information_decomposition_v10_7_m`

## Status

New semantic diagnostic patch. This is not a bug fix and does not replace V10.7_l.

## Entry script

```text
stage_partition/V10/v10.7/scripts/run_window_anchor_information_decomposition_v10_7_m.py
```

## Source module

```text
stage_partition/V10/v10.7/src/stage_partition_v10_7/window_anchor_information_decomposition_pipeline.py
```

## Output directory

```text
stage_partition/V10/v10.7/outputs/window_anchor_information_decomposition_v10_7_m
```

## Execution chain

```text
run_window_anchor_information_decomposition_v10_7_m.py
  -> Settings(...)
  -> run_window_anchor_information_decomposition_v10_7_m(settings)
     -> load smoothed_fields.npz
     -> build H/P daily metrics under raw/anomaly/local_background_removed modes
     -> Q2 information-form decomposition
     -> Q1/Q4 E2 anchor specificity
     -> Q3 scalarized-vs-signed representation comparison
     -> Q5-min metric specificity
     -> tables / figures / run_meta / markdown summary
```

## Questions answered

```text
Q2: Which H_zonal_width information form is effective?
Q1/Q4: Is E2/W33 an effective anchor for the H -> P narrow channel?
Q3: Does scalarized transition representation lose signed structural-component information?
Q5-min: Is the route source/target metric specific inside H -> P?
```

## Not answered

```text
- Causal effect.
- Full W33 -> W45 mapping.
- Full P/V/H/Je/Jw object network.
- Whether transition windows represent most object information.
```
