# V10.7_b H W045 Gaussian derivative scale-space diagnostic

## 1. Purpose and boundary

This is a dedicated scale diagnostic on the H object state matrix around W045.
It is **not** a `ruptures.Window` rerun and must not be treated as detector-width sensitivity.
It does not infer causality, physical mechanism, yearwise stability, or spatial continuity.
Its role is heuristic: identify whether H19/H35/H45/H57 appear as scale-space transition-energy structures and help choose targets for later yearwise/spatial tests.

## 2. Settings

- sigmas: [2, 3, 4, 5, 7, 9, 11, 14]
- focus day range: 10–62
- target days: {'H19': 19, 'H35': 35, 'H45': 45, 'H57': 57}
- scale backend: scipy_gaussian_filter1d
- used H features: 11 / input features: 11

## 3. Ridge-family summary

| ridge_id   |   day_min |   day_max |   day_center_weighted |   sigma_min |   sigma_max |   persistence_fraction | nearest_target_label   |   nearest_target_distance | role_hint                 |
|:-----------|----------:|----------:|----------------------:|------------:|------------:|-----------------------:|:-----------------------|--------------------------:|:--------------------------|
| R001       |        18 |        26 |               20.3773 |           2 |          14 |                  1     | H19                    |                         1 | candidate_scale_structure |
| R002       |        35 |        36 |               35.6846 |           2 |           4 |                  0.375 | H35                    |                         1 | candidate_scale_structure |

## 4. Target-day scale identity hints

| target_label   |   target_day | nearest_ridge_id   | scale_identity_hint                             |   persistence_fraction |   max_energy_norm | recommended_next_step_target                                      |
|:---------------|-------------:|:-------------------|:------------------------------------------------|-----------------------:|------------------:|:------------------------------------------------------------------|
| H19            |           19 | R001               | stable_or_candidate_cross_scale_structure       |                  1     |          1        | H19 or H19-H35 package if linked to H35                           |
| H35            |           35 | R002               | medium_persistence_candidate_local_bump         |                  0.375 |          0.798908 | H35 local structure plus E2 integrated H strength                 |
| H45            |           45 |                    | no_clear_scale_structure_near_W045_main_cluster |                nan     |          0.808178 | supports H absence in W045 main-cluster at scale-diagnostic layer |
| H57            |           57 |                    | weak_or_unclear_post_W045_structure             |                nan     |          0.789041 | H57 as post-W045 reference if ridge is persistent                 |

**Global H35 note:** H35 has a separate scale structure; it may be tested heuristically but is not a confirmed weak precursor.

## 5. Interpretation rules

- If H35 shares a ridge with H19, later tests should use an H19–H35 prewindow package rather than H35 alone.
- If H35 forms a separate medium-persistence ridge, it can be retained as a heuristic E2 target, but still not as a confirmed weak precursor.
- If H45 lacks a clear ridge, that supports H absence in the W045 main-cluster at the scale-diagnostic layer only.
- If H57 forms a ridge, it should be treated as a post-W045 reference candidate, not automatically as W045 response.

## 6. Recommended next step

- H35 next-step target: H35 local structure plus E2 integrated H strength
- H45 implication: supports H absence in W045 main-cluster at scale-diagnostic layer
- Do not convert these scale hints into physical interpretation until yearwise and spatial-field checks are run.
