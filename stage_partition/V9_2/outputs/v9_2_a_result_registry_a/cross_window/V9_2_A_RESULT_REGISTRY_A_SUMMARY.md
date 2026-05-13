# V9.2_a Result Registry A Summary
## 1. Scope
This is a read-only candidate-result registry for V9.2_a direct-year 2D spatiotemporal MVEOF outputs. It does not rerun MVEOF, does not rerun peak detection, and does not perform physical interpretation.
## 2. Evidence status
- Evidence stage: candidate PC-group composite timing result.
- Group-peak leave-one-year stability: marked as not_run_or_incomplete unless a non-empty input table is present.
- Single-year influence audit: not audited in this registry.
- Physical interpretation: not started.

## 3. Candidate level definitions
- Candidate-A: clear high/low timing difference by object peak shift or pair order change.
- Candidate-B: moderate high/low timing difference.
- Candidate-C: field mode exists but timing difference is weak.
- Candidate-Hold: detector quality problem.

## 4. Window-mode registry overview
- W045 mode1: Candidate-A_clear_high_low_timing_difference; high: H(38) -> Je(44)/Jw(45)/V(47)/P(48); low: H(33) -> Jw(41)/V(43)/P(45)/Je(46); order_changes=3; max_object_shift=5.0.
- W045 mode2: Candidate-A_clear_high_low_timing_difference; high: Jw(39) -> Je(45)/V(46)/H(47)/P(47); low: H(44)/V(44)/P(47) -> Jw(49)/Je(50); order_changes=6; max_object_shift=10.0.
- W045 mode3: Candidate-A_clear_high_low_timing_difference; high: H(31) -> Jw(40)/P(42) -> V(46)/Je(48); low: V(37)/P(39)/Jw(41) -> H(43) -> Je(49); order_changes=6; max_object_shift=12.0.
- W081 mode1: Candidate-A_clear_high_low_timing_difference; high: Je(69) -> H(83)/Jw(83)/V(85) -> P(88); low: Je(73)/H(75)/V(75)/P(78) -> Jw(86); order_changes=6; max_object_shift=10.0.
- W081 mode2: Candidate-A_clear_high_low_timing_difference; high: V(70) -> Jw(79)/Je(82)/H(83)/P(83); low: H(75)/Je(76)/P(78)/V(80) -> Jw(83); order_changes=7; max_object_shift=10.0.
- W081 mode3: Candidate-A_clear_high_low_timing_difference; high: Je(69)/Jw(73) -> H(76)/P(80)/V(80); low: H(76)/V(76)/Je(79)/Jw(81)/P(82); order_changes=8; max_object_shift=10.0.
- W113 mode1: Candidate-A_clear_high_low_timing_difference; high: Jw(108)/V(108)/Je(111) -> P(113)/H(114); low: Jw(103) -> Je(109) -> P(113)/V(114)/H(116); order_changes=7; max_object_shift=6.0.
- W113 mode2: Candidate-A_clear_high_low_timing_difference; high: Jw(101)/V(101)/P(104) -> H(109) -> Je(118); low: V(105) -> P(112)/Jw(113)/Je(113) -> H(118); order_changes=5; max_object_shift=12.0.
- W113 mode3: Candidate-A_clear_high_low_timing_difference; high: Jw(104)/Je(104) -> H(109)/V(110) -> P(118); low: Jw(113)/V(114)/Je(115)/P(116)/H(117); order_changes=7; max_object_shift=11.0.
- W160 mode1: Candidate-A_clear_high_low_timing_difference; high: Jw(148)/V(148) -> P(154) -> Je(165)/H(167); low: H(153) -> Je(159)/P(160)/V(161)/Jw(163); order_changes=9; max_object_shift=15.0.
- W160 mode2: Candidate-A_clear_high_low_timing_difference; high: Je(155)/H(157)/P(158)/V(158) -> Jw(166); low: Jw(151) -> H(157) -> V(162)/Je(165)/P(165); order_changes=7; max_object_shift=15.0.
- W160 mode3: Candidate-A_clear_high_low_timing_difference; high: V(155)/Jw(157)/P(159)/H(160) -> Je(166); low: Jw(149) -> H(156) -> Je(161)/P(161)/V(165); order_changes=7; max_object_shift=10.0.

## 5. Prohibited interpretation
- Do not treat PC high/low as physical year types.
- Do not treat candidate levels as final robust tiers.
- Do not treat group-composite peak differences as single-year rules.
- Do not interpret MVEOF modes as physical mechanisms before field/profile audit.

## 6. Recommended next use
Use `v9_2_a_window_mode_registry.csv` and `v9_2_a_window_mode_sequence_summary.csv` to extract candidate timing patterns by window-mode. Use detector and missing-evidence columns to avoid overclaiming.
