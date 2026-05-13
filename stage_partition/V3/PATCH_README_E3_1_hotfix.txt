E-3.1 hotfix

Fixes traceback:
AttributeError: 'ObjectProfile' object has no attribute 'profile'

Cause:
E-3 point_significance.py incorrectly assumed profile_dict['P'] exposed a `.profile` Series.
In the V3 codebase, ObjectProfile stores raw_cube / lat_grid / masks, while the detector profile
is available from rw_out['profile'] in pipeline.py.

Fix:
1. run_point_null_significance now accepts `observed_profile: pd.Series`
2. pipeline.py passes `rw_out['profile']` into run_point_null_significance

Apply into:
D:\easm_project01
Then rerun:
python D:\easm_project01\stage_partition\V3\scripts\run_stage_partition_v3.py
