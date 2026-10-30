"""Command-line entry points for preparing and running the experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .data import load_recipes
from .embeddings import generate_embeddings, load_embedding_cache
from .experiment import ExperimentOptions, run_experiment
from .splits import DEFAULT_PROPORTIONS, PreparedDataset, SplitPlan, make_outer_splits


def _default_data(project_root: Path) -> Path:
    return project_root / "data" / "whats-cooking" / "train.json"


def _add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, required=True, help="labelled What's Cooking train.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="COMP 345 cuisine-classification experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("prepare-splits", help="freeze the outer and validation split manifest")
    _add_common_dataset_args(split_parser)
    split_parser.add_argument("--output", type=Path, required=True)
    split_parser.add_argument("--seed", type=int, default=345)
    split_parser.add_argument("--outer-folds", type=int, default=3)
    split_parser.add_argument("--validation-fraction", type=float, default=0.15)

    embedding_parser = subparsers.add_parser("generate-embeddings", help="cache frozen Gemini embeddings")
    _add_common_dataset_args(embedding_parser)
    embedding_parser.add_argument("--output", type=Path, required=True)
    embedding_parser.add_argument("--api-key", default=None)
    embedding_parser.add_argument("--request-delay", type=float, default=1.0)
    embedding_parser.add_argument("--max-retries", type=int, default=2)
    embedding_parser.add_argument("--batch-size", type=int, default=100, help="recipes per API request; maximum 100")
    embedding_parser.add_argument("--items-per-minute", type=int, default=1000, help="local input rate cap; maximum 3000")
    embedding_parser.add_argument("--start-index", type=int, default=0, help="inclusive recipe index for this segment")
    embedding_parser.add_argument("--end-index", type=int, default=None, help="exclusive recipe index; defaults to 100 recipes per run")

    run_parser = subparsers.add_parser("run-experiment", help="run all four registered methods")
    _add_common_dataset_args(run_parser)
    run_parser.add_argument("--splits", type=Path, required=True)
    run_parser.add_argument("--embeddings", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--seed", type=int, default=345)
    run_parser.add_argument("--proportions", type=int, nargs="+", default=list(DEFAULT_PROPORTIONS))
    run_parser.add_argument("--quick", action="store_true", help="small registered grid for a smoke run")
    run_parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    run_parser.add_argument("--skip-probability-fusion", action="store_true")
    run_parser.add_argument("--skip-feature-fusion", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare-splits":
        dataset = PreparedDataset.from_recipes(load_recipes(args.data))
        plan = make_outer_splits(
            dataset,
            seed=args.seed,
            n_splits=args.outer_folds,
            validation_fraction=args.validation_fraction,
        )
        plan.save(args.output)
        print(f"wrote {args.output} ({dataset.n_samples} recipes, {len(plan.folds)} outer folds)")
        return 0
    if args.command == "generate-embeddings":
        recipes = load_recipes(args.data)
        end_index = args.end_index if args.end_index is not None else min(args.start_index + 100, len(recipes))
        try:
            output = generate_embeddings(
                recipes,
                args.output,
                api_key=args.api_key,
                request_delay=args.request_delay,
                max_retries=args.max_retries,
                batch_size=args.batch_size,
                items_per_minute=args.items_per_minute,
                start_index=args.start_index,
                end_index=end_index,
            )
        except RuntimeError as exc:
            parser.error(str(exc))
        if output.exists():
            print(f"completed embedding cache: {output}")
        else:
            print(
                f"saved progress for indices [{args.start_index}, {end_index}); "
                f"rerun the next segment with the same output path: {args.output}"
            )
        return 0
    if args.command == "run-experiment":
        dataset = PreparedDataset.from_recipes(load_recipes(args.data))
        plan = SplitPlan.load(args.splits, dataset)
        embeddings = load_embedding_cache(args.embeddings, dataset.recipes)
        options = ExperimentOptions(
            seed=args.seed,
            quick=args.quick,
            proportions=tuple(args.proportions),
            bootstrap_replicates=args.bootstrap_replicates,
            skip_probability_fusion=args.skip_probability_fusion,
            skip_feature_fusion=args.skip_feature_fusion,
        )
        result = run_experiment(dataset, plan, embeddings, args.output_dir, options=options)
        print(f"wrote {args.output_dir / 'results.json'} ({len(result['rows'])} evaluation rows)")
        return 0
    parser.error(f"unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
