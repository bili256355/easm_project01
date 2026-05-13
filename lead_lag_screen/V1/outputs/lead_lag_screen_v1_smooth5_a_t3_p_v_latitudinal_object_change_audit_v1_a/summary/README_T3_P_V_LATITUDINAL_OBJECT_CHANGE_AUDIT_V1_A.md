# T3 P/V850 Latitudinal Object-Change Audit v1_a

This layer diagnoses **object-level changes only** for precipitation and v850.
It does not compute or read V->P support/R2/pathway evidence.

## Scope

- P mean state and change maps.
- V850 mean state and change maps.
- P latitudinal profiles, multi-band detection, and cross-window band links.
- V850 raw / positive / absolute latitudinal profiles and positive north-edge diagnostics.
- Lat-band summaries using the actual data latitude range.

## Default windows

{'S3': (87, 106), 'T3_early': (107, 112), 'T3_full': (107, 117), 'T3_late': (113, 117), 'S4': (118, 154)}

## Key restriction

Do not use this output to claim V causes/explains P. This output only states how
P and V850 objects themselves change across S3 -> T3 -> S4.
