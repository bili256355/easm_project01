# T3 V→P transition-chain report v1_b

This layer is a reporting / evidence-chain layer. It does **not** rerun V1,
does **not** rerun bootstrap, and does **not** write full-grid NPZ map outputs.

## Purpose

Connect three evidence layers through S3 → T3_early → T3_full → T3_late → S4:

1. precipitation object-state and object-change maps/tables;
2. v850 object-state and object-change maps/tables;
3. V→P support transition tables and support-change maps.

All statements about increase/decrease/shift must be tied to an explicit
comparison, region, and V component.

## Default windows

{'S3': (87, 106), 'T3_early': (107, 112), 'T3_full': (107, 117), 'T3_late': (113, 117), 'S4': (118, 154)}

## Output directory

`D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_t3_v_to_p_transition_chain_report_v1_b`
