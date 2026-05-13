# UPDATE_LOG_V7_Z_CLEAN

## v7_z_clean

Purpose: extract a clean W45 multi-object pre-post mainline after V7-z/hotfix/Je-audit iterations.

Main decisions implemented:

1. Raw/profile object-state vector is the only main object-window detector input.
2. Daily unit-norm shape-pattern detector is not used as a main object-window input.
3. Pattern remains in pre-post extraction through R_diff / S_pattern.
4. Paired year bootstrap is used for object-window and pairwise curve metrics.
5. Evidence-family summary and co-transition veto prevent repeated weak tendencies from becoming lead claims.
6. Output is written to a new clean directory; V7-v/w/x/y/z outputs are not overwritten.

Not included:

- shape-pattern object-window mainline;
- Je audit a/b/c reruns;
- distribution-pattern branch;
- causality/pathway/PCMCI/lead-lag.
