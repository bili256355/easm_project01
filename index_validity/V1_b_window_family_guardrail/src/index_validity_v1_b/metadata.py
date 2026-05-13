from __future__ import annotations

from typing import Dict, List, Tuple

VARIABLE_ORDER: List[str] = [
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

FAMILY_ORDER: List[str] = ["P", "V", "H", "Je", "Jw"]

REGIONS: Dict[str, Dict[str, object]] = {
    "P": {"field": "precip", "lon_range": (105.0, 125.0), "lat_range": (10.0, 50.0)},
    "V": {"field": "v850", "lon_range": (105.0, 125.0), "lat_range": (10.0, 30.0)},
    "H": {"field": "z500", "lon_range": (110.0, 140.0), "lat_range": (10.0, 40.0)},
    "Je": {"field": "u200", "lon_range": (120.0, 150.0), "lat_range": (20.0, 50.0)},
    "Jw": {"field": "u200", "lon_range": (80.0, 110.0), "lat_range": (20.0, 50.0)},
}

INDEX_METADATA: Dict[str, Dict[str, object]] = {
    "P_main_band_share": {"family": "P", "field": "precip", "kind": "band_share", "expected_type": "main_gt_other", "expected_meaning": "Main rain-band share: Main(24-35N)/Total(10-40N)", "band_lines": [10, 18, 24, 35, 40, 45, 50]},
    "P_south_band_share_18_24": {"family": "P", "field": "precip", "kind": "band_share", "expected_type": "south_gt_main", "expected_meaning": "South rain-band share: South(18-24N)/Total(10-40N)", "band_lines": [10, 18, 24, 35, 40, 45, 50]},
    "P_main_minus_south": {"family": "P", "field": "precip", "kind": "band_difference", "expected_type": "main_gt_south", "expected_meaning": "Main-band minus south-band share", "band_lines": [10, 18, 24, 35, 40, 45, 50]},
    "P_spread_lat": {"family": "P", "field": "precip", "kind": "spread", "expected_type": "spread_like", "expected_meaning": "Latitudinal spread of precipitation over 10-40N", "band_lines": [10, 18, 24, 35, 40, 45, 50]},
    "P_north_band_share_35_45": {"family": "P", "field": "precip", "kind": "band_share", "expected_type": "north_gt_main", "expected_meaning": "North rain-band share: North(35-45N)/Total(10-50N)", "band_lines": [10, 18, 24, 35, 40, 45, 50]},
    "P_north_minus_main_35_45": {"family": "P", "field": "precip", "kind": "band_difference", "expected_type": "north_gt_main", "expected_meaning": "North-band share minus main-band share", "band_lines": [10, 18, 24, 35, 40, 45, 50]},
    "P_total_centroid_lat_10_50": {"family": "P", "field": "precip", "kind": "position", "expected_type": "north_gt_south", "expected_meaning": "Precipitation centroid latitude over 10-50N", "band_lines": [10, 18, 24, 35, 40, 45, 50]},
    "V_strength": {"family": "V", "field": "v850", "kind": "strength", "expected_type": "domain_mean_positive", "expected_meaning": "Mean V over 10-30N", "band_lines": [10, 15, 30]},
    "V_pos_centroid_lat": {"family": "V", "field": "v850", "kind": "position", "expected_type": "north_gt_south", "expected_meaning": "Positive-v850 centroid latitude over 10-30N", "band_lines": [10, 15, 30]},
    "V_NS_diff": {"family": "V", "field": "v850", "kind": "lat_difference", "expected_type": "north_gt_south_v", "expected_meaning": "North minus south V: sum 15-30N minus sum 10-15N", "band_lines": [10, 15, 30]},
    "H_strength": {"family": "H", "field": "z500", "kind": "strength", "expected_type": "domain_mean_positive", "expected_meaning": "H/z500 core strength", "band_lines": []},
    "H_centroid_lat": {"family": "H", "field": "z500", "kind": "position", "expected_type": "north_gt_south", "expected_meaning": "Latitude centroid of H core", "band_lines": []},
    "H_west_extent_lon": {"family": "H", "field": "z500", "kind": "extent", "expected_type": "generic_pattern", "expected_meaning": "Western extent longitude of H core; direction is definition-sensitive", "band_lines": []},
    "H_zonal_width": {"family": "H", "field": "z500", "kind": "width", "expected_type": "generic_pattern", "expected_meaning": "Zonal width of H core; boundary-sensitive", "band_lines": []},
    "Je_strength": {"family": "Je", "field": "u200", "kind": "strength", "expected_type": "domain_mean_positive", "expected_meaning": "Downstream jet strength", "band_lines": [20, 50]},
    "Je_axis_lat": {"family": "Je", "field": "u200", "kind": "position", "expected_type": "north_gt_south", "expected_meaning": "Downstream jet axis latitude", "band_lines": [20, 50]},
    "Je_meridional_width": {"family": "Je", "field": "u200", "kind": "width", "expected_type": "generic_pattern", "expected_meaning": "Downstream jet meridional width", "band_lines": [20, 50]},
    "Jw_strength": {"family": "Jw", "field": "u200", "kind": "strength", "expected_type": "domain_mean_positive", "expected_meaning": "Upstream jet strength", "band_lines": [20, 50]},
    "Jw_axis_lat": {"family": "Jw", "field": "u200", "kind": "position", "expected_type": "north_gt_south", "expected_meaning": "Upstream jet axis latitude", "band_lines": [20, 50]},
    "Jw_meridional_width": {"family": "Jw", "field": "u200", "kind": "width", "expected_type": "generic_pattern", "expected_meaning": "Upstream jet meridional width", "band_lines": [20, 50]},
}
