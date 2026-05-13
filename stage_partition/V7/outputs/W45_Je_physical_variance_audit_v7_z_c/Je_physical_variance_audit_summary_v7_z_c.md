# Je Physical Variance Audit for W45 (V7-z-c)

## Purpose
Test whether the Je day30-34 low shape-norm episode can be physically described as a decrease in spatial variance / flattened profile structure, rather than only as a numerical normalization artifact.

## Final status
- `profile_and_2D_variance_dip_supported`

## Key interpretation
- The day30-34 period has physical support as a low spatial-variance / flattened-structure episode in Je. This supports the interpretation that the shape-pattern peak is amplified by a real low-variance background, not produced from nothing.

## Window means
- early_pre_day20_29 (20-29): profile_std=6.095, region2d_std=5.949
- early_core_day26_33 (26-33): profile_std=5.482, region2d_std=5.457
- low_variance_day30_34 (30-34): profile_std=5.265, region2d_std=5.241
- early_post_day35_39 (35-39): profile_std=6.268, region2d_std=6.077
- W45_day40_48 (40-48): profile_std=7.576, region2d_std=7.422
- late_main_day40_52 (40-52): profile_std=7.613, region2d_std=7.448

## Bootstrap highlights
- low_vs_W45 / profile_spatial_std: median delta=-2.332, q025=-3.417, q975=-1.157, P(first<second)=1.000, decision=first_lower_supported
- low_vs_W45 / region2d_spatial_std: median delta=-2.188, q025=-3.321, q975=-1.031, P(first<second)=1.000, decision=first_lower_supported
- low_vs_post / profile_spatial_std: median delta=-1.017, q025=-1.996, q975=-0.02654, P(first<second)=0.979, decision=first_lower_supported
- low_vs_post / region2d_spatial_std: median delta=-0.8442, q025=-1.781, q975=0.08183, P(first<second)=0.958, decision=first_lower_tendency
- low_vs_pre / profile_spatial_std: median delta=-0.8429, q025=-1.835, q975=0.08086, P(first<second)=0.962, decision=first_lower_tendency
- low_vs_pre / region2d_spatial_std: median delta=-0.6958, q025=-1.627, q975=0.1859, P(first<second)=0.927, decision=first_lower_tendency

## Minimum-day bootstrap
- profile_spatial_std: P(min in day30-34)=0.881, median min day=31.0, decision=minimum_day30_34_supported
- region2d_spatial_std: P(min in day30-34)=0.836, median min day=31.0, decision=minimum_day30_34_supported

## Allowed statement
- Je day30-34 may be described as a low spatial-variance / flattened-profile episode only if the final status is supported or observed-profile-only; otherwise keep it as an unresolved hypothesis.

## Forbidden statement
- Do not claim this proves Je causally drives W45 transitions. Do not claim shape-pattern day33 is a confirmed main physical transition equal to the robust raw/profile day46 peak.