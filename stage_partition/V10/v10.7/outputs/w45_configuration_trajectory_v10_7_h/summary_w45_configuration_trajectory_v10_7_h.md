# V10.7_h W45 configuration trajectory audit

## Method boundary
- This audit treats W45 as a multi-object configuration made by P/V/H/Je/Jw.
- It does not control P/V/Je/Jw away as covariates.
- It tests within-year E1/E2/M configuration coupling against shuffled-year null.
- It is not causal inference and does not prove physical pathways.

## Input status
| object   | status   | source_key      | reducer          | source_note                                        |
|:---------|:---------|:----------------|:-----------------|:---------------------------------------------------|
| P        | loaded   | precip_smoothed | spatial_rms      | precip object proxy domain                         |
| V        | loaded   | v850_smoothed   | spatial_rms      | v850 object proxy domain                           |
| H        | loaded   | z500_smoothed   | spatial_rms      | H object domain                                    |
| Je       | loaded   | u200_smoothed   | jet_q90_strength | derived from u200 eastern sector: 120-150E, 25-45N |
| Jw       | loaded   | u200_smoothed   | jet_q90_strength | derived from u200 western sector: 80-110E, 25-45N  |

## Primary coupling decisions (anomaly, cosine)
| pair   |   observed_mean_similarity |   null_p90_similarity |   permutation_p | decision                  |
|:-------|---------------------------:|----------------------:|----------------:|:--------------------------|
| E1_E2  |                  0.229842  |            0.132415   |      0.00199601 | strong_coupled_above_null |
| E2_M   |                 -0.0311415 |            0.0578792  |      0.487026   | not_above_null            |
| E1_M   |                 -0.14431   |           -0.00865614 |      0.806387   | not_above_null            |

## E2-M object contribution
| object_removed   |   observed_drop_mean |   null_drop_p90 |   permutation_p | contribution_class               |
|:-----------------|---------------------:|----------------:|----------------:|:---------------------------------|
| P                |           -0.0635355 |      -0.0237094 |        0.670659 | negative_or_disruptive_dimension |
| V                |           -0.0318751 |       0.038714  |        0.9002   | negative_or_disruptive_dimension |
| H                |            0.0663849 |       0.0686462 |        0.117764 | secondary_dimension              |
| Je               |            0.014686  |       0.0561674 |        0.43513  | secondary_dimension              |
| Jw               |            0.0481838 |       0.0560955 |        0.137725 | secondary_dimension              |

## Route decision
| decision_item                   | status                                 | evidence                                                                                                                       | route_implication                                                                                                                               |
|:--------------------------------|:---------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|
| E2_to_M_configuration_coupling  | not_above_null                         | obs=-0.03114; null_p90=0.05788; p=0.487                                                                                        | E2 multi-object activity should not be treated as organized W45 preconfiguration under this metric/mode.                                        |
| E1_to_E2_preconfiguration       | strong_coupled_above_null              | obs=0.2298; null_p90=0.1324; p=0.001996                                                                                        | If coupled, early pre-window multi-object activity may form an organized preconfiguration sequence; if not, E1 should not be over-linked to E2. |
| E1_to_M_direct_link             | not_above_null                         | obs=-0.1443; null_p90=-0.008656; p=0.8064                                                                                      | Direct E1-M coupling would support a longer pre-window trajectory; absence suggests E1 cannot be directly tied to M without E2 evidence.        |
| H_contribution_to_E2_M_coupling | secondary_dimension                    | drop=0.06638; null_p90=0.06865; p=0.1178                                                                                       | H contributes weakly/secondarily; retain only as a secondary E2 component, not a standalone precursor.                                          |
| W45_configuration_route         | E2_not_supported_as_M_preconfiguration | primary_mode=anomaly; primary_metric=cosine; E2_M=not_above_null; H_contribution=secondary_dimension; positive_objects=H,Jw,Je | Do not interpret E1/E2 activity, including H35, as W45 preconfiguration without further evidence.                                               |

## Forbidden interpretations
- Do not interpret object contribution as causality.
- Do not treat E1/E2 object activity as W45 preconfiguration unless coupling exceeds shuffled-year null.
- Do not treat H as absent from W45 formation merely because H has no synchronous H45 peak.