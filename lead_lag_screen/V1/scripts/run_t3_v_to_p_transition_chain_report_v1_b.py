# -*- coding: utf-8 -*-
r"""Run T3 V->P transition-chain report v1_b.

Default command:
    python D:\easm_project01\lead_lag_screen\V1\scripts\run_t3_v_to_p_transition_chain_report_v1_b.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _infer_project_root() -> Path:
    # Expected script path: D:/easm_project01/lead_lag_screen/V1/scripts/...
    return Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T3 V->P transition-chain report v1_b.")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--previous-output-dir", type=str, default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-cartopy", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else _infer_project_root()
    src_dir = project_root / "lead_lag_screen" / "V1" / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from lead_lag_screen_v1.t3_v_to_p_transition_chain_settings import TransitionChainReportSettings
    from lead_lag_screen_v1.t3_v_to_p_transition_chain_pipeline import run_t3_v_to_p_transition_chain_report_v1_b

    settings = TransitionChainReportSettings(project_root=project_root)
    if args.output_dir:
        settings.output_dir = Path(args.output_dir)
    if args.previous_output_dir:
        settings.previous_field_explanation_output_dir = Path(args.previous_output_dir)
    if args.no_figures:
        settings.make_figures = False
    if args.no_cartopy:
        settings.use_cartopy = False

    summary = run_t3_v_to_p_transition_chain_report_v1_b(settings)
    print("[DONE] T3 V->P transition-chain report v1_b complete.")
    print(f"Output: {summary['output_dir']}")


if __name__ == "__main__":
    main()
