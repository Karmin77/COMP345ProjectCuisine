"""Run the local TF-IDF baseline without a Gemini API cache."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from comp345.baseline import run_tfidf_baseline  # noqa: E402
from comp345.data import load_recipes  # noqa: E402
from comp345.splits import DEFAULT_PROPORTIONS, PreparedDataset, SplitPlan  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=345)
    parser.add_argument("--proportions", type=int, nargs="+", default=list(DEFAULT_PROPORTIONS))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    dataset = PreparedDataset.from_recipes(load_recipes(args.data))
    plan = SplitPlan.load(args.splits, dataset)
    result = run_tfidf_baseline(
        dataset,
        plan,
        args.output_dir,
        seed=args.seed,
        proportions=tuple(args.proportions),
        quick=args.quick,
    )
    print(f"wrote {args.output_dir / 'results.json'} ({len(result['rows'])} evaluation rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
