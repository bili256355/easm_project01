# UPDATE_LOG_V7_N

## w45_allfield_process_relation_layer_v7_n

Purpose: replace the old single-marker implementation layer for W45 with a process-level relation layer for P/V/H/Je/Jw.

This patch adds an implementation-layer audit that represents:

- each field as a pre-to-post progress process, not one timing day;
- each pair as daily curve relation + phase relation + marker-family relation;
- W45 as a multi-layer organization object, not a forced order chain.

Key constraints:

- Uses current V7-e/V7-m progress representation.
- Does not switch to raw025.
- Does not change W45 window, pre/post periods, progress definition, thresholds, or spatial units.
- Does not infer causality.
- Does not exclude any of P/V/H/Je/Jw from computation.
- Does not interpret not-resolved as synchrony.

Outputs are written to:

`outputs/w45_allfield_process_relation_layer_v7_n`

Run with:

`python D:\easm_project01\stage_partition\V7\scripts\run_w45_allfield_process_relation_layer_v7_n.py`
