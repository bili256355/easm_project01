# UPDATE_LOG - V10.1 joint main-window reproduction

## v10.1 initial patch

- Added contained subproject under `V10/v10.1/`.
- Semantically rewrites the V6 -> V6_1 joint-object discovery chain without importing old stage_partition modules.
- Rebuilds joint state matrix, ruptures.Window detector output, local candidates, bootstrap support, candidate bands, derived windows, uncertainty summaries, and strict accepted-window lineage mapping.
- Reads V6/V6_1 CSVs only for regression audit.
- Does not run object-native peak discovery, sensitivity testing, pair-order analysis, or physical interpretation.
