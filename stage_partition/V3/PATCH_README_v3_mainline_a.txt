E-3 patch: refine strict point significance local comparator.

Main changes:
- local comparator now selects the best matched peak near each formal point
- matched statistic is computed against a local background window
- point_null_significance now reports p_match_exist and p_match_exceed_conditional
- strict local interpretation is based on matched statistics rather than arbitrary local peaks
- output tag defaults to mainline_v3_h
