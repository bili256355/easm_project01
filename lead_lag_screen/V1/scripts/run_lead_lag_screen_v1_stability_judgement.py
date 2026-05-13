from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
V1_ROOT = THIS_FILE.parents[1]
SRC = V1_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lead_lag_screen_v1.stability_judgement_settings import V1StabilityJudgementSettings
from lead_lag_screen_v1.stability_judgement_pipeline import run_v1_stability_judgement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process lead_lag_screen/V1 outputs into lag-vs-tau0 stability judgement labels."
    )
    parser.add_argument("--project-root", type=Path, default=Path(r"D:\easm_project01"))
    parser.add_argument("--input-tag", default="lead_lag_screen_v1_smooth5_a")
    parser.add_argument("--output-tag", default="lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b")
    parser.add_argument("--ci-level", choices=["90", "95"], default="90")
    parser.add_argument("--p-lag-gt-tau0-threshold", type=float, default=0.90)
    parser.add_argument("--p-forward-gt-reverse-threshold", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = V1StabilityJudgementSettings(
        project_root=args.project_root,
        input_tag=args.input_tag,
        output_tag=args.output_tag,
        ci_level=args.ci_level,
        p_lag_gt_tau0_threshold=args.p_lag_gt_tau0_threshold,
        p_forward_gt_reverse_threshold=args.p_forward_gt_reverse_threshold,
    )
    summary = run_v1_stability_judgement(settings)
    print("V1 stability judgement completed.")
    print(f"Input:  {summary['input_dir']}")
    print(f"Output: {summary['output_dir']}")
    print(f"Rows:   {summary['n_rows']}")
    print("Judgement counts:")
    for k, v in summary["stability_judgement_counts"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
