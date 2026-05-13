from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowRelation:
    overlap_days: int
    union_days: int
    overlap_ratio: float
    center_distance_days: int
    near_peak_flag: bool
    overlap_any_flag: bool
    strict_overlap_flag: bool
    relation_tier: str


def compute_window_overlap(candidate_start: int, candidate_end: int, ref_start: int, ref_end: int) -> tuple[int, int, float]:
    overlap_days = max(0, min(int(candidate_end), int(ref_end)) - max(int(candidate_start), int(ref_start)) + 1)
    union_days = max(int(candidate_end), int(ref_end)) - min(int(candidate_start), int(ref_start)) + 1
    overlap_ratio = overlap_days / float(max(1, union_days))
    return int(overlap_days), int(union_days), float(overlap_ratio)


def compute_center_distance(candidate_center: int, ref_center: int) -> int:
    return abs(int(candidate_center) - int(ref_center))


def classify_window_relation(
    candidate_start: int,
    candidate_end: int,
    candidate_center: int,
    ref_start: int,
    ref_end: int,
    ref_center: int,
    *,
    near_peak_tolerance_days: int,
    overlap_days_min: int,
    overlap_ratio_min: float,
) -> WindowRelation:
    overlap_days, union_days, overlap_ratio = compute_window_overlap(candidate_start, candidate_end, ref_start, ref_end)
    center_distance_days = compute_center_distance(candidate_center, ref_center)
    near_peak_flag = center_distance_days <= int(near_peak_tolerance_days)
    overlap_any_flag = overlap_days >= 1
    strict_overlap_flag = (overlap_days >= int(overlap_days_min)) and (overlap_ratio >= float(overlap_ratio_min))
    if strict_overlap_flag:
        relation_tier = 'strict_overlap'
    elif overlap_any_flag:
        relation_tier = 'overlap_any'
    elif near_peak_flag:
        relation_tier = 'near_peak_only'
    else:
        relation_tier = 'none'
    return WindowRelation(
        overlap_days=int(overlap_days),
        union_days=int(union_days),
        overlap_ratio=float(overlap_ratio),
        center_distance_days=int(center_distance_days),
        near_peak_flag=bool(near_peak_flag),
        overlap_any_flag=bool(overlap_any_flag),
        strict_overlap_flag=bool(strict_overlap_flag),
        relation_tier=relation_tier,
    )
