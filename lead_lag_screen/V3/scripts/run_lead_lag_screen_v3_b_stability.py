from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    here = Path(__file__).resolve()
    src = here.parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    _add_src_to_path()
    from lead_lag_screen_v3.settings import LeadLagScreenV3Settings
    from lead_lag_screen_v3b.settings_b import StabilitySettings
    from lead_lag_screen_v3b.pipeline_stability import run_lead_lag_screen_v3_b_stability

    parser = argparse.ArgumentParser(description="Run lead_lag_screen/V3_b EOF-PC1 formal stability judgement.")
    parser.add_argument("--project-root", default=r"D:\easm_project01")
    parser.add_argument("--output-tag", default="eof_pc1_smooth5_v3_b_stability_quick_a")
    parser.add_argument("--relation-bootstrap", type=int, default=1000)
    parser.add_argument("--mode-bootstrap", type=int, default=30)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=100)
    parser.add_argument("--formal-mode-bootstrap", action="store_true", help="Use 500 PC1-mode bootstrap replicates for a heavier formal mode-stability run.")
    args = parser.parse_args()

    base = LeadLagScreenV3Settings(
        project_root=Path(args.project_root),
        output_tag=args.output_tag,
    )
    mode_bootstrap = 500 if args.formal_mode_bootstrap else args.mode_bootstrap
    stability = StabilitySettings(
        output_tag=args.output_tag,
        n_relation_bootstrap=args.relation_bootstrap,
        n_pc1_mode_bootstrap=mode_bootstrap,
        bootstrap_chunk_size=args.bootstrap_chunk_size,
    )
    summary = run_lead_lag_screen_v3_b_stability(base_settings=base, stability_settings=stability)
    print(summary)


if __name__ == "__main__":
    main()
