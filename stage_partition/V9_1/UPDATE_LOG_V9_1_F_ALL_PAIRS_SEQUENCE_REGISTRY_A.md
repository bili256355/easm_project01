# UPDATE_LOG V9.1_f_all_pairs_sequence_registry_a

Purpose: add a read-only target-conditioned window-level sequence registry for V9.1_f_all_pairs_a.

This layer does not rerun bootstrap, MCA/SVD, or peak detection. It merges each target's score phase with precomputed bootstrap object peaks, so each target high/low phase can be summarized as a full P/V/H/Je/Jw peak sequence rather than only its target-pair order.

Interpretation boundary: sequences are target-conditioned bootstrap-space summaries; they are not physical regimes, direct-year types, or causal mechanisms.
