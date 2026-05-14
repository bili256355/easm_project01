from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def _best_target_ridge(label: str, target_response: pd.DataFrame, ridge_summary: pd.DataFrame) -> dict[str, Any]:
    resp = target_response[target_response["target_label"] == label].copy() if target_response is not None else pd.DataFrame()
    if resp.empty:
        return {"target_label": label, "ridge_id": "", "status": "no_response_rows"}
    ridge_ids = [r for r in resp.get("nearest_ridge_id", pd.Series(dtype=str)).astype(str).tolist() if r]
    if not ridge_ids:
        # No linked local ridge. Use scale response only.
        return {
            "target_label": label,
            "ridge_id": "",
            "status": "no_linked_ridge",
            "max_energy_norm": float(resp["energy_norm"].max()) if "energy_norm" in resp else np.nan,
            "max_energy_percentile": float(resp["energy_percentile"].max()) if "energy_percentile" in resp else np.nan,
            "n_sigmas_with_nearby_ridge": 0,
        }
    counts = pd.Series(ridge_ids).value_counts()
    best_ridge = str(counts.index[0])
    rs = ridge_summary[ridge_summary["ridge_id"].astype(str) == best_ridge].copy() if ridge_summary is not None else pd.DataFrame()
    if rs.empty:
        return {
            "target_label": label,
            "ridge_id": best_ridge,
            "status": "linked_ridge_missing_summary",
            "n_sigmas_with_nearby_ridge": int(counts.iloc[0]),
        }
    row = rs.iloc[0].to_dict()
    row.update({"target_label": label, "ridge_id": best_ridge, "status": "linked_ridge", "n_sigmas_with_nearby_ridge": int(counts.iloc[0])})
    return row


def classify_h_w045_scale_identity(
    ridge_summary: pd.DataFrame,
    target_response: pd.DataFrame,
    target_days: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target_records = {label: _best_target_ridge(label, target_response, ridge_summary) for label in target_days}
    h19 = target_records.get("H19", {})
    h35 = target_records.get("H35", {})
    h45 = target_records.get("H45", {})
    h57 = target_records.get("H57", {})

    same_19_35 = bool(h19.get("ridge_id") and h19.get("ridge_id") == h35.get("ridge_id"))
    h19_persist = float(h19.get("persistence_fraction", 0.0) or 0.0)
    h35_persist = float(h35.get("persistence_fraction", 0.0) or 0.0)
    h45_persist = float(h45.get("persistence_fraction", 0.0) or 0.0)
    h57_persist = float(h57.get("persistence_fraction", 0.0) or 0.0)

    # H19
    if h19.get("status") == "linked_ridge" and h19_persist >= 0.50:
        h19_status = "stable_or_candidate_cross_scale_structure"
    elif h19.get("status") == "linked_ridge":
        h19_status = "limited_persistence_structure"
    else:
        h19_status = "no_clear_linked_ridge"
    rows.append({
        "target_label": "H19",
        "target_day": int(target_days.get("H19", 19)),
        "nearest_ridge_id": h19.get("ridge_id", ""),
        "scale_identity_hint": h19_status,
        "persistence_fraction": h19.get("persistence_fraction", np.nan),
        "max_energy_norm": h19.get("max_energy_norm", np.nan),
        "recommended_next_step_target": "H19 or H19-H35 package if linked to H35",
        "interpretation_limit": "scale diagnostic only; not physical mechanism",
    })

    # H35
    if same_19_35:
        h35_status = "same_ridge_as_H19_broad_prewindow_package_candidate"
        target = "H19-H35 prewindow package"
    elif h35.get("status") == "linked_ridge" and 0.30 <= h35_persist < 0.60:
        h35_status = "medium_persistence_candidate_local_bump"
        target = "H35 local structure plus E2 integrated H strength"
    elif h35.get("status") == "linked_ridge" and h35_persist >= 0.60:
        h35_status = "stable_independent_scale_structure_candidate"
        target = "H35 as candidate event for yearwise/spatial test"
    else:
        h35_status = "weak_or_no_clear_scale_structure"
        target = "do not prioritize H35 single-day target"
    rows.append({
        "target_label": "H35",
        "target_day": int(target_days.get("H35", 35)),
        "nearest_ridge_id": h35.get("ridge_id", ""),
        "scale_identity_hint": h35_status,
        "persistence_fraction": h35.get("persistence_fraction", np.nan),
        "max_energy_norm": h35.get("max_energy_norm", np.nan),
        "recommended_next_step_target": target,
        "interpretation_limit": "does not prove weak precursor or conditioning role",
    })

    # H45
    if h45.get("status") == "linked_ridge" and h45_persist >= 0.30:
        h45_status = "some_scale_structure_near_W045_main_cluster"
        target = "check H45 before claiming H absence"
    else:
        h45_status = "no_clear_scale_structure_near_W045_main_cluster"
        target = "supports H absence in W045 main-cluster at scale-diagnostic layer"
    rows.append({
        "target_label": "H45",
        "target_day": int(target_days.get("H45", 45)),
        "nearest_ridge_id": h45.get("ridge_id", ""),
        "scale_identity_hint": h45_status,
        "persistence_fraction": h45.get("persistence_fraction", np.nan),
        "max_energy_norm": h45.get("max_energy_norm", np.nan),
        "recommended_next_step_target": target,
        "interpretation_limit": "H absence remains method/scale diagnostic until spatial/yearwise checks",
    })

    # H57
    if h57.get("status") == "linked_ridge" and h57_persist >= 0.30:
        h57_status = "post_W045_scale_structure_candidate"
    else:
        h57_status = "weak_or_unclear_post_W045_structure"
    rows.append({
        "target_label": "H57",
        "target_day": int(target_days.get("H57", 57)),
        "nearest_ridge_id": h57.get("ridge_id", ""),
        "scale_identity_hint": h57_status,
        "persistence_fraction": h57.get("persistence_fraction", np.nan),
        "max_energy_norm": h57.get("max_energy_norm", np.nan),
        "recommended_next_step_target": "H57 as post-W045 reference if ridge is persistent",
        "interpretation_limit": "not automatically W045 response",
    })

    global_note = ""
    if same_19_35:
        global_note = "H19 and H35 share one linked ridge; use H19-H35 package rather than H35 alone."
    elif h35_status in {"medium_persistence_candidate_local_bump", "stable_independent_scale_structure_candidate"}:
        global_note = "H35 has a separate scale structure; it may be tested heuristically but is not a confirmed weak precursor."
    else:
        global_note = "H35 has weak/unclear scale structure; avoid making it a central event target."

    out = pd.DataFrame(rows)
    out["global_h35_note"] = global_note
    return out
