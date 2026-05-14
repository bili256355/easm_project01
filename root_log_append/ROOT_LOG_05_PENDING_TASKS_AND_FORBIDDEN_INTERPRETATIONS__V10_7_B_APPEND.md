# ROOT_LOG_05 append: V10.7_b interpretation boundary

## New pending task opened by V10.7_b

V10.7_b is intended to identify scale-space clues around H19/H35/H45/H57. Its outputs should decide the target of later yearwise/spatial checks:

- If H35 is an independent ridge: later tests may target H35 as a heuristic local structure.
- If H35 is linked to H19: later tests should use an H19-H35 prewindow package rather than H35 alone.
- If H45 has no ridge: this supports H absence from the W045 main cluster only at the scale-diagnostic layer.
- If H57 has a ridge: treat it as a post-W045 reference candidate, not automatically as a W045 response.

## Forbidden interpretations

Do not write:

- `detector_width sensitivity proves the true physical scale of H35`.
- `V10.7_b proves H35 is a weak precursor`.
- `V10.7_b proves H35 provides background conditions`.
- `V10.7_b is a spatial or yearwise validation`.
- `Gaussian sigma is equivalent to the original ruptures detector_width`.

Allowed wording:

- `V10.7_b provides a dedicated scale-space diagnostic on the H object state matrix.`
- `It gives heuristic evidence for selecting later yearwise/spatial targets.`
- `It should not be converted into physical mechanism claims without additional evidence.`
