#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run T3 V→P field-explanation hard-evidence audit v1_a.

This entry DOES NOT rerun the V1 lead-lag screen and DOES NOT modify older
T3 physical-hypothesis audit outputs. It adds an independent hard-evidence
layer that evaluates V-index variability against the precipitation field.

Default output:
lead_lag_screen/V1/outputs/lead_lag_screen_v1_smooth5_a_t3_v_to_p_field_explanation_audit_v1_a
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

    from lead_lag_screen_v1.t3_v_to_p_field_explanation_settings import (
        FieldExplanationAuditSettings,
    )
    from lead_lag_screen_v1.t3_v_to_p_field_explanation_pipeline import (
        run_t3_v_to_p_field_explanation_audit,
    )

    parser = argparse.ArgumentParser(
        description="Run T3 V→P field-explanation hard-evidence audit v1_a."
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260429)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--no-cartopy",
        action="store_true",
        help="Force plain lon/lat figures instead of the default cartopy map backend.",
    )
    parser.add_argument(
        "--use-legacy-t3-window",
        action="store_true",
        help=(
            "Use legacy physical-audit T3 windows (S3 90-107, T3 106-120, S4 120-158). "
            "Default is the V6_1/V1 main-screen window basis (S3 87-106, T3 107-117, S4 118-154)."
        ),
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = (
            args.project_root
            / "lead_lag_screen"
            / "V1"
            / "outputs"
            / "lead_lag_screen_v1_smooth5_a_t3_v_to_p_field_explanation_audit_v1_a"
        )

    settings = FieldExplanationAuditSettings(
        project_root=args.project_root,
        output_dir=out_dir,
        max_lag=args.max_lag,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        make_figures=not args.no_figures,
        use_cartopy=not args.no_cartopy,
        use_legacy_t3_window=args.use_legacy_t3_window,
    )
    summary = run_t3_v_to_p_field_explanation_audit(settings)
    print("T3 V→P field-explanation hard-evidence audit completed.")
    print(f"Output directory: {settings.output_dir}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
