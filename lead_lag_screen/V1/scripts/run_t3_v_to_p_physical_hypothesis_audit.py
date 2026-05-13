#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run T3 V→P physical hypothesis audit.

This is a V1-side audit utility. It does NOT rerun the V1 lead-lag screen.
It reads existing V1 / V1 stability / index_validity outputs plus smooth5 fields
and produces hypothesis-oriented diagnostics for why T3 V→P fixed-index
relations contract.

Default output:
lead_lag_screen/V1/outputs/lead_lag_screen_v1_smooth5_a_t3_v_to_p_physical_hypothesis_audit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _infer_project_root() -> Path:
    # .../lead_lag_screen/V1/scripts/run_*.py -> project root
    return Path(__file__).resolve().parents[3]


def main() -> None:
    project_root = _infer_project_root()
    src_dir = project_root / "lead_lag_screen" / "V1" / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from lead_lag_screen_v1.t3_v_to_p_physical_hypothesis_pipeline import (
        PhysicalHypothesisAuditSettings,
        run_t3_v_to_p_physical_hypothesis_audit,
    )

    parser = argparse.ArgumentParser(
        description="Audit physical hypotheses for T3 V→P contraction."
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-cartopy", action="store_true")
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument(
        "--south-scs-lon-max",
        type=float,
        default=130.0,
        help="Longitude max for South China / SCS diagnostic region.",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = (
            args.project_root
            / "lead_lag_screen"
            / "V1"
            / "outputs"
            / "lead_lag_screen_v1_smooth5_a_t3_v_to_p_physical_hypothesis_audit"
        )

    settings = PhysicalHypothesisAuditSettings(
        project_root=args.project_root,
        output_dir=out_dir,
        make_figures=not args.no_figures,
        use_cartopy=not args.no_cartopy,
        max_lag=args.max_lag,
        south_scs_lon_max=args.south_scs_lon_max,
    )
    summary = run_t3_v_to_p_physical_hypothesis_audit(settings)
    print("T3 V→P physical hypothesis audit completed.")
    print(f"Output directory: {settings.output_dir}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
