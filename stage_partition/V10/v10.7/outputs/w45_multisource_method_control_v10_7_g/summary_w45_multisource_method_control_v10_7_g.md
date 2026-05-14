# V10.7_g W45 multisource method-control audit

## Method boundary
- This is a multisource yearwise incremental-association audit for W45.
- Je/Jw are derived from u200 sectors, not searched as independent fields.
- Self-target results are diagnostic only; cross-target results are the main evidence.
- Positive results indicate yearwise incremental association, not causality.

## Input audit
object status      source_key lat_range   lon_range          reducer                                        source_note note  n_years  n_days  n_lat  n_lon  added_year_axis
     H loaded   z500_smoothed 15.0-35.0 110.0-140.0      spatial_rms                                    H object domain            45     183     81    121            False
     P loaded precip_smoothed 15.0-35.0 110.0-140.0      spatial_rms                         precip object proxy domain            45     183     81    121            False
     V loaded   v850_smoothed 15.0-35.0 110.0-140.0      spatial_rms                           v850 object proxy domain            45     183     81    121            False
    Je loaded   u200_smoothed 25.0-45.0 120.0-150.0 jet_q90_strength derived from u200 eastern sector: 120-150E, 25-45N            45     183     81    121            False
    Jw loaded   u200_smoothed 25.0-45.0  80.0-110.0 jet_q90_strength  derived from u200 western sector: 80-110E, 25-45N            45     183     81    121            False

## Method-control decision
 decision_item                                    status                                       evidence                                                                              route_implication
method control method_has_detectable_cross_object_signal Non-H cross-target supports: clear=28, weak=2. Use this to decide whether H negative results are meaningful or whether the audit lacks power.
     H package        H_package_has_cross_object_support       H clear=1, weak=0 cross-target supports.                        Keep H package as preconditioning candidate for supported targets only.

## Object route decision
source_object                         status                                                                                                                                                                                                                                                    evidence                                                   route_implication
            H keep_as_cross_object_candidate                                                                                                                                                                                                                                             pre_package->Jw Eligible for gated spatial high/low composite on supported targets.
            P keep_as_cross_object_candidate                 main_cluster_source->V; main_cluster_source->Je; main_cluster_source->M; main_cluster_source->joint_proxy; main_cluster_source->M_minus_P; main_cluster_source->M_minus_V; main_cluster_source->M_minus_Je; main_cluster_source->M_minus_Jw Eligible for gated spatial high/low composite on supported targets.
            V keep_as_cross_object_candidate                 main_cluster_source->P; main_cluster_source->Je; main_cluster_source->M; main_cluster_source->joint_proxy; main_cluster_source->M_minus_P; main_cluster_source->M_minus_V; main_cluster_source->M_minus_Je; main_cluster_source->M_minus_Jw Eligible for gated spatial high/low composite on supported targets.
           Je keep_as_cross_object_candidate pre_package->Jw; main_cluster_source->P; main_cluster_source->V; main_cluster_source->M; main_cluster_source->joint_proxy; main_cluster_source->M_minus_P; main_cluster_source->M_minus_V; main_cluster_source->M_minus_Je; main_cluster_source->M_minus_Jw Eligible for gated spatial high/low composite on supported targets.
           Jw keep_as_cross_object_candidate                                                                                                                                                             main_cluster_source->M_minus_P; main_cluster_source->M_minus_V; main_cluster_source->M_minus_Je Eligible for gated spatial high/low composite on supported targets.

## Primary-mode top cross-target incremental records
source_object source_package_type target_object           target_type  delta_r2  delta_cv_rmse_positive_means_improvement  source_coef_loo_sign_stability  permutation_p_delta_r2                  decision
            P main_cluster_source    M_minus_Jw  leave_one_object_out  0.710899                                  0.539401                             1.0                0.009901 clear_incremental_support
            V main_cluster_source    M_minus_Jw  leave_one_object_out  0.672375                                  0.459083                             1.0                0.009901 clear_incremental_support
            V main_cluster_source    M_minus_Je  leave_one_object_out  0.632650                                  0.450707                             1.0                0.009901 clear_incremental_support
            V main_cluster_source             M    combined_available  0.571889                                  0.384424                             1.0                0.009901 clear_incremental_support
            V main_cluster_source   joint_proxy joint_proxy_available  0.571889                                  0.384424                             1.0                0.009901 clear_incremental_support
            P main_cluster_source    M_minus_Je  leave_one_object_out  0.558008                                  0.400419                             1.0                0.009901 clear_incremental_support
            P main_cluster_source             M    combined_available  0.548143                                  0.400593                             1.0                0.009901 clear_incremental_support
            P main_cluster_source   joint_proxy joint_proxy_available  0.548143                                  0.400593                             1.0                0.009901 clear_incremental_support
            V main_cluster_source             P       individual_main  0.495063                                  0.290238                             1.0                0.009901 clear_incremental_support
            P main_cluster_source             V       individual_main  0.478543                                  0.309520                             1.0                0.009901 clear_incremental_support
           Je main_cluster_source    M_minus_Jw  leave_one_object_out  0.470972                                  0.300403                             1.0                0.009901 clear_incremental_support
           Je main_cluster_source     M_minus_V  leave_one_object_out  0.456834                                  0.301596                             1.0                0.009901 clear_incremental_support
            V main_cluster_source     M_minus_P  leave_one_object_out  0.438641                                  0.268538                             1.0                0.009901 clear_incremental_support
            P main_cluster_source     M_minus_V  leave_one_object_out  0.418826                                  0.280999                             1.0                0.009901 clear_incremental_support
           Je main_cluster_source     M_minus_P  leave_one_object_out  0.385904                                  0.240475                             1.0                0.009901 clear_incremental_support
           Je main_cluster_source   joint_proxy joint_proxy_available  0.382568                                  0.228840                             1.0                0.009901 clear_incremental_support
           Je main_cluster_source             M    combined_available  0.382568                                  0.228840                             1.0                0.009901 clear_incremental_support
            V main_cluster_source     M_minus_V  leave_one_object_out  0.271967                                  0.149535                             1.0                0.009901 clear_incremental_support
            P main_cluster_source     M_minus_P  leave_one_object_out  0.243834                                  0.149398                             1.0                0.009901 clear_incremental_support
           Je main_cluster_source             P       individual_main  0.179047                                  0.078472                             1.0                0.009901 clear_incremental_support
           Je         pre_package            Jw       individual_main  0.170145                                  0.071780                             1.0                0.009901 clear_incremental_support
            P main_cluster_source            Je       individual_main  0.161408                                  0.054149                             1.0                0.009901 clear_incremental_support
           Jw main_cluster_source     M_minus_P  leave_one_object_out  0.152861                                  0.082118                             1.0                0.029703 clear_incremental_support
           Jw main_cluster_source     M_minus_V  leave_one_object_out  0.121344                                  0.058998                             1.0                0.009901 clear_incremental_support
            V main_cluster_source            Je       individual_main  0.118382                                  0.030578                             1.0                0.009901 clear_incremental_support
           Jw main_cluster_source    M_minus_Je  leave_one_object_out  0.117051                                  0.059354                             1.0                0.019802 clear_incremental_support
           Je main_cluster_source    M_minus_Je  leave_one_object_out  0.105038                                  0.026266                             1.0                0.019802 clear_incremental_support
           Je main_cluster_source             V       individual_main  0.101735                                  0.025114                             1.0                0.019802 clear_incremental_support
            V         pre_package            Je       individual_main  0.072954                                 -0.013130                             1.0                0.108911    no_incremental_support
            H         pre_package            Jw       individual_main  0.059958                                  0.020680                             1.0                0.089109 clear_incremental_support