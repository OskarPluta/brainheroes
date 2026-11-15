"""Command-line entry point: `uv run brainheroes <command>`."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .constants import DATA_DIR, DOCS_DIR, RESULTS_DIR


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="brainheroes", description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="raw dataset root (default: %(default)s)")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR, help="results directory (default: %(default)s)")
    parser.add_argument("--docs", type=Path, default=DOCS_DIR, help="generated reports (default: %(default)s)")
    sub = parser.add_subparsers(dest="command", required=True)
    feats = sub.add_parser("features", help="extract feature tables from raw SNIRF files (~3 min)")
    feats.add_argument("--force", action="store_true", help="re-extract even if the CSVs exist")
    sub.add_parser("score", help="two-flag scores for both cohorts -> results/two_flag_*")
    report = sub.add_parser("report", help="render docs/REPORT.html and docs/REPORT.md")
    report.add_argument("--fragment", type=Path, help="also write a body-only HTML fragment here")
    sub.add_parser("benchmark", help="off-the-shelf detectors vs. the two-flag rule -> docs/SIMPLE_DETECTORS.md")
    sub.add_parser("ablation", help="earlier 50/50 variant and single-score ablations -> results/verification_summary.json")
    sub.add_parser("all", help="features (if missing), score, report, benchmark, ablation")
    args = parser.parse_args(argv)

    # Imports are deferred so `--help` stays fast and matplotlib loads only for the ablation.
    steps = ["features", "score", "report", "benchmark", "ablation"] if args.command == "all" else [args.command]
    for step in steps:
        start = time.perf_counter()
        if step == "features":
            from .features import build_feature_tables

            build_feature_tables(args.data, args.results, force=getattr(args, "force", False))
        elif step == "score":
            from . import two_flag

            summary = two_flag.run(args.results)
            for key in ("supine", "supine_crit_b", "dot_crit_a", "dot"):
                b = summary[key]
                print(f"{key:14s} AUC {b['auc']:.3f}  alarms: {b['tp']}/{b['n_pos']} hit, {b['fp']} false (n={b['n']})")
        elif step == "report":
            from . import report as report_mod

            report_mod.run(args.results, args.docs, fragment=getattr(args, "fragment", None))
        elif step == "benchmark":
            from . import detectors

            detectors.run(args.results, args.docs)
        elif step == "ablation":
            from . import ablation

            summary = ablation.run(args.results)
            print(json.dumps(summary["young_cohort_id38_audit"], indent=2))
        print(f"[{step}] done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
