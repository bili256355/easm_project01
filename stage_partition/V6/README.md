# stage_partition/V6

V6-a is a point-level screening branch.

It does only three things:
1. build a registry of all baseline detected local peaks,
2. compute bootstrap support as the headline significance metric,
3. compute yearwise local-peak support as auxiliary context.

V6-a does **not** include window judgement, competition, parameter-path, final judgement,
or object-aware support. Formal primary status is kept only as an annotation field and is not a gate for inclusion.


Current default mainline contract
--------------------------------
- output_tag: mainline_v6_a
- bootstrap replicates: 1000
- bootstrap strict/match/near windows: 2 / 5 / 8 days
- yearwise support is auxiliary only; current windows are aligned to 2 / 5 / 8 days for consistency
