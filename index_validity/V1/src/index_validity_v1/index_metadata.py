\
from __future__ import annotations

VARIABLE_ORDER = [
    "P_main_band_share",
    "P_south_band_share_18_24",
    "P_main_minus_south",
    "P_spread_lat",
    "P_north_band_share_35_45",
    "P_north_minus_main_35_45",
    "P_total_centroid_lat_10_50",
    "V_strength",
    "V_pos_centroid_lat",
    "V_NS_diff",
    "H_strength",
    "H_centroid_lat",
    "H_west_extent_lon",
    "H_zonal_width",
    "Je_strength",
    "Je_axis_lat",
    "Je_meridional_width",
    "Jw_strength",
    "Jw_axis_lat",
    "Jw_meridional_width",
]

REGIONS = {
    "P": {"field": "precip", "lon_range": (105.0, 125.0), "lat_range": (10.0, 50.0)},
    "V": {"field": "v850", "lon_range": (105.0, 125.0), "lat_range": (10.0, 30.0)},
    "H": {"field": "z500", "lon_range": (110.0, 140.0), "lat_range": (10.0, 40.0)},
    "Je": {"field": "u200", "lon_range": (120.0, 150.0), "lat_range": (20.0, 50.0)},
    "Jw": {"field": "u200", "lon_range": (80.0, 110.0), "lat_range": (20.0, 50.0)},
}

INDEX_METADATA = {
    "P_main_band_share": {
        "family": "P", "field": "precip", "physical_check_type": "lat_profile_and_map",
        "expected_meaning": "Main rain-band share: Main(24-35N) / Total_P(10-40N)",
        "band_lines": [18, 24, 35, 40, 45, 50],
    },
    "P_south_band_share_18_24": {
        "family": "P", "field": "precip", "physical_check_type": "lat_profile_and_map",
        "expected_meaning": "South rain-band share: South(18-24N) / Total_P(10-40N)",
        "band_lines": [18, 24, 35, 40, 45, 50],
    },
    "P_main_minus_south": {
        "family": "P", "field": "precip", "physical_check_type": "lat_profile_and_map",
        "expected_meaning": "Main-band minus south-band share",
        "band_lines": [18, 24, 35, 40, 45, 50],
    },
    "P_spread_lat": {
        "family": "P", "field": "precip", "physical_check_type": "lat_profile_and_map",
        "expected_meaning": "Latitudinal spread of precipitation over 10-40N",
        "band_lines": [10, 18, 24, 35, 40, 45, 50],
    },
    "P_north_band_share_35_45": {
        "family": "P", "field": "precip", "physical_check_type": "lat_profile_and_map",
        "expected_meaning": "North rain-band share: North(35-45N) / Total_P_late(10-50N)",
        "band_lines": [10, 18, 24, 35, 40, 45, 50],
    },
    "P_north_minus_main_35_45": {
        "family": "P", "field": "precip", "physical_check_type": "lat_profile_and_map",
        "expected_meaning": "North-band share minus main-band share over 10-50N denominator",
        "band_lines": [10, 18, 24, 35, 40, 45, 50],
    },
    "P_total_centroid_lat_10_50": {
        "family": "P", "field": "precip", "physical_check_type": "lat_profile_and_map",
        "expected_meaning": "Precipitation centroid latitude over 10-50N",
        "band_lines": [10, 18, 24, 35, 40, 45, 50],
    },
    "V_strength": {
        "family": "V", "field": "v850", "physical_check_type": "map_and_lat_profile",
        "expected_meaning": "Mean V over 10-30N",
        "band_lines": [10, 15, 30],
    },
    "V_pos_centroid_lat": {
        "family": "V", "field": "v850", "physical_check_type": "map_and_lat_profile",
        "expected_meaning": "Positive-v850 centroid latitude over 10-30N",
        "band_lines": [10, 15, 30],
    },
    "V_NS_diff": {
        "family": "V", "field": "v850", "physical_check_type": "map_and_lat_profile",
        "expected_meaning": "North minus south V structure: sum 15-30N minus sum 10-15N",
        "band_lines": [10, 15, 30],
    },
    "H_strength": {
        "family": "H", "field": "z500", "physical_check_type": "map",
        "expected_meaning": "High-core z500 strength in H region",
        "band_lines": [],
    },
    "H_centroid_lat": {
        "family": "H", "field": "z500", "physical_check_type": "map",
        "expected_meaning": "Latitude centroid of thresholded H core",
        "band_lines": [],
    },
    "H_west_extent_lon": {
        "family": "H", "field": "z500", "physical_check_type": "map",
        "expected_meaning": "Western extent longitude of thresholded H core",
        "band_lines": [],
    },
    "H_zonal_width": {
        "family": "H", "field": "z500", "physical_check_type": "map",
        "expected_meaning": "Zonal width of thresholded H core",
        "band_lines": [],
    },
    "Je_strength": {
        "family": "Je", "field": "u200", "physical_check_type": "map_and_lat_profile",
        "expected_meaning": "Downstream jet strength",
        "band_lines": [20, 50],
    },
    "Je_axis_lat": {
        "family": "Je", "field": "u200", "physical_check_type": "map_and_lat_profile",
        "expected_meaning": "Downstream jet axis latitude",
        "band_lines": [20, 50],
    },
    "Je_meridional_width": {
        "family": "Je", "field": "u200", "physical_check_type": "map_and_lat_profile",
        "expected_meaning": "Downstream jet meridional width",
        "band_lines": [20, 50],
    },
    "Jw_strength": {
        "family": "Jw", "field": "u200", "physical_check_type": "map_and_lat_profile",
        "expected_meaning": "Upstream jet strength",
        "band_lines": [20, 50],
    },
    "Jw_axis_lat": {
        "family": "Jw", "field": "u200", "physical_check_type": "map_and_lat_profile",
        "expected_meaning": "Upstream jet axis latitude",
        "band_lines": [20, 50],
    },
    "Jw_meridional_width": {
        "family": "Jw", "field": "u200", "physical_check_type": "map_and_lat_profile",
        "expected_meaning": "Upstream jet meridional width",
        "band_lines": [20, 50],
    },
}
