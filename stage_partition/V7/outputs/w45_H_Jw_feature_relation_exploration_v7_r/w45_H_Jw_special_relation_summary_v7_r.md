# W45 H/Jw feature relation exploration V7-r

## 1. Purpose
This diagnostic branch reduces the scope to H and Jw and explores feature-level evidence for their W45 relation.
It reads V7-q feature-level outputs and is not a clean trunk rebuild.

## 2. Input boundary
- Input V7-q dir: `D:\easm_project01\stage_partition\V7\outputs\w45_feature_process_resolution_v7_q`
- Uses V7-m/n/o/p outputs as input: false
- Uses V7-q outputs as diagnostic input: true

## 3. Feature provenance
- Physical region interpretation allowed: False
- If false, all feature-coordinate patterns are feature-index diagnostics only, not regional conclusions.

## 4. Departure same-phase evidence
- departure90: delta Jw-H=0, near_equal=0.864, label=strong_same_phase_candidate
- departure95: delta Jw-H=0, near_equal=0.864, label=strong_same_phase_candidate
- t10: delta Jw-H=2, near_equal=0.136, label=same_phase_not_established

## 5. H early-progress support
- t10: H earlier fraction=1, weighted=1, label=broad_feature_support
- t15: H earlier fraction=1, weighted=1, label=broad_feature_support
- t20: H earlier fraction=1, weighted=1, label=broad_feature_support
- t25: H earlier fraction=1, weighted=1, label=broad_feature_support
- t30: H earlier fraction=1, weighted=1, label=broad_feature_support
- t35: H earlier fraction=1, weighted=1, label=broad_feature_support
- t40: H earlier fraction=0.818, weighted=0.894, label=broad_feature_support
- t45: H earlier fraction=0.818, weighted=0.894, label=broad_feature_support
- t50: H earlier fraction=0.727, weighted=0.808, label=broad_feature_support

## 6. Jw catch-up / finish support
- t50: Jw support fraction=0.0909, weighted=0.0449, label=limited_feature_catchup
- t75: Jw support fraction=0.364, weighted=0.279, label=limited_feature_catchup
- peak_smooth3: Jw support fraction=0, weighted=0, label=no_feature_catchup_support
- duration_25_75: Jw support fraction=0.636, weighted=0.625, label=broad_feature_catchup_support
- tail_50_75: Jw support fraction=0.545, weighted=0.326, label=core_feature_catchup_support
- early_span_25_50: Jw support fraction=0, weighted=0, label=no_feature_catchup_support

## 7. Distribution overlap by phase
- departure_phase: delta=0, overlap=1, label=strong_overlap
- early_progress_phase: delta=2, overlap=0, label=H_shifted_earlier
- mid_progress_phase: delta=5, overlap=0.0139, label=H_shifted_earlier
- finish_phase: delta=-1, overlap=0.553, label=moderate_overlap
- peak_phase: delta=6, overlap=0, label=H_shifted_earlier

## 8. Overall diagnostic card
- Overall relation candidate: same_departure_with_H_frontloaded_and_Jw_catchup_support

## 9. Prohibited interpretations
- Do not write H globally leads Jw from this branch.
- Do not write H/Jw are synchronous; near-same-phase is only a candidate without an equivalence test.
- Do not infer physical regions from feature IDs unless provenance allows coordinate interpretation.
- Do not delete weak/noisy features; they are retained but down-weighted.
