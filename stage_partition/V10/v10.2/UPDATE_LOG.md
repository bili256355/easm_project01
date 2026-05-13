# UPDATE_LOG: V10.2 object-native peak discovery

## v10.2 initial patch

Purpose: add an object-native full-season peak discovery layer after V10.1 successfully reproduced the joint-object main-window discovery lineage.

Design constraints:

- contained under `V10/v10.2/`;
- does not write to V10 main `scripts/src/outputs/logs`;
- does not import V6/V6_1/V7/V9 modules;
- reads V10.1 lineage and V10 main peak/subpeak outputs only as mapping references;
- does not perform sensitivity testing or physical interpretation.
