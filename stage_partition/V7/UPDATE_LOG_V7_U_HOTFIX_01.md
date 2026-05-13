# UPDATE_LOG_V7_U_HOTFIX_01

## Patch

`v7_u_hotfix_01_departure_after_pre`

## Purpose

Fix the V7-u departure-event definition. The original V7-u allowed `departure_from_pre` to be detected inside the same pre-period used to estimate the pre-state envelope. This produced invalid or contaminated events such as departure at day30/day37 and made the departure layer unsuitable for H/Jw order/synchrony adjudication.

## Changes

1. `departure_from_pre` search now starts strictly after `pre_period_end`:

```text
search_start = pre_period_end + 1
```

2. `departure_from_pre` is retained only as a non-durable candidate diagnostic:

```text
candidate_non_durable_after_pre
```

3. `durable_departure_from_pre_3d` is the main departure event for integrated interpretation.

4. `input_audit_v7_u.json` now records:

```text
departure_search_start_day
departure_search_rule
departure_raw_first_crossing_including_pre
departure_candidate_after_pre
durable_departure_after_pre_3d
```

5. `run_meta.json` now records:

```text
hotfix_id = v7_u_hotfix_01_departure_after_pre
departure_hotfix_rule = departure search starts after pre_period_end; durable_departure_from_pre_3d is the main departure event; non-durable departure_from_pre is candidate-only
```

## Scope

This is a hotfix. It does not change:

- state/growth separation logic;
- H/Jw-only scope;
- bootstrap rule;
- state/post-dominance definitions;
- growth metric definitions;
- output directory name.

## Run

```bash
cd D:\easm_project01
python stage_partition\V7\scripts\run_w45_H_Jw_state_growth_transition_framework_v7_u.py
```

Debug:

```bash
cd D:\easm_project01
set V7U_DEBUG_N_BOOTSTRAP=20
set V7U_SKIP_FIGURES=1
python stage_partition\V7\scripts\run_w45_H_Jw_state_growth_transition_framework_v7_u.py
```

## Interpretation

After this hotfix, pre-period crossings are no longer valid departure events. If `durable_departure_from_pre_3d` still fails to produce a hard H/Jw order/synchrony decision, the departure layer should be treated as method-unclosed rather than rescued by the old non-durable event.
