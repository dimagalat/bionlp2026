#!/usr/bin/env python3
"""
Deliberative Council — DMRS Defense-Level Classifier
=====================================================

Run the multi-phase council pipeline on the shared task data.

Usage:

    # Run predictions (requires GOOGLE_API_KEY env var or --api-key)
    uv run python run.py --test-path data/test.json --train-path data/train.json

    # Evaluate on training data (reports performance)
    uv run python run.py --train-path data/train.json --eval-on-train --limit 50

    # Evaluate predictions against gold labels (no API calls)
    uv run python run.py evaluate --gold data/train.json --pred prediction.json

    # Override models
    uv run python run.py --model gemini-2.5-flash --moderator-model gemini-2.5-pro
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

from src.config import CouncilConfig, ModelConfig
from src.council import Council
from src.evaluate import compute_metrics, format_report, format_distribution
from src.retriever import ExampleRetriever


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Deliberative Council — DMRS Defense-Level Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = root.add_subparsers(dest="command")

    pred = sub.add_parser("predict", help="Run council predictions (default)")
    _add_predict_args(pred)
    _add_predict_args(root)

    ev = sub.add_parser("evaluate", help="Evaluate predictions against gold labels")
    ev.add_argument("--gold", required=True, help="Path to gold-standard JSON")
    ev.add_argument("--pred", required=True, help="Path to prediction JSON")
    ev.add_argument("--title", default="Evaluation", help="Report title")

    return root


def _add_predict_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--api-key", default=None,
                   help="Google API key (or set GOOGLE_API_KEY env var)")
    p.add_argument("--model", default="gemini-2.5-flash",
                   help="Model for specialist agents (default: gemini-2.5-flash)")
    p.add_argument("--moderator-model", default="gemini-2.5-pro",
                   help="Model for resolution phases (default: gemini-2.5-pro)")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="Temperature for specialist agents (default: 0.1)")

    p.add_argument("--train-path", default="data/train.json",
                   help="Path to training data (for few-shot retrieval)")
    p.add_argument("--test-path", default="data/test.json",
                   help="Path to test data")
    p.add_argument("--output-path", default="prediction.json",
                   help="Where to write predictions (default: prediction.json)")
    p.add_argument("--log-path", default=None,
                   help="Path to write detailed per-sample reasoning log")

    p.add_argument("--num-few-shot", type=int, default=5,
                   help="Few-shot examples per query (default: 5)")
    p.add_argument("--num-candidates", type=int, default=30,
                   help="TF-IDF candidate pool size before MMR (default: 30)")
    p.add_argument("--proto-weight", type=float, default=0.35,
                   help="Prototypicality vs query-similarity weight (default: 0.35)")
    p.add_argument("--mmr-lambda", type=float, default=0.7,
                   help="MMR: 1.0=relevance, 0.0=diversity (default: 0.7)")

    p.add_argument("--eval-on-train", action="store_true",
                   help="Use train data as test; run predictions and report performance")
    p.add_argument("--concurrent-samples", type=int, default=5,
                   help="Max samples processed concurrently (default: 5)")
    p.add_argument("--max-parallel-agents", type=int, default=5,
                   help="Max agents per sample running concurrently (default: 5)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N test samples")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable debug logging")


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

async def run_predict(args: argparse.Namespace) -> None:
    if getattr(args, "eval_on_train", False):
        args.test_path = args.train_path

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("ERROR: Google API key required. Set GOOGLE_API_KEY or use --api-key.")

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("council")
    logger.info("Model: %s  |  Moderator: %s", args.model, args.moderator_model)

    # --- Load data ---
    train_data = json.loads(Path(args.train_path).read_text())
    test_data = json.loads(Path(args.test_path).read_text())
    if args.limit:
        test_data = test_data[: args.limit]
    logger.info("Train: %d samples  |  Test: %d samples", len(train_data), len(test_data))
    has_labels = "label" in test_data[0] if test_data else False

    # --- Build retriever ---
    logger.info("Building retriever (candidates=%d, proto=%.2f, mmr_λ=%.2f)",
                args.num_candidates, args.proto_weight, args.mmr_lambda)
    retriever = ExampleRetriever(
        train_data,
        num_candidates=args.num_candidates,
        proto_weight=args.proto_weight,
        mmr_lambda=args.mmr_lambda,
    )

    # --- Configure council ---
    default_model = ModelConfig(
        provider="google", model=args.model,
        api_key=api_key, temperature=args.temperature,
    )
    moderator_model = ModelConfig(
        provider="google", model=args.moderator_model,
        api_key=api_key, temperature=0.1,
    )
    config = CouncilConfig(
        default_model=default_model,
        moderator_model=moderator_model,
        num_few_shot=args.num_few_shot,
        num_candidates=args.num_candidates,
        max_parallel_agents=args.max_parallel_agents,
    )
    council = Council(config, retriever)

    # --- Run predictions ---
    logger.info("Running council predictions ...")
    sem = asyncio.Semaphore(args.concurrent_samples)
    pbar = tqdm(total=len(test_data), desc="Council", file=sys.stderr)

    async def _process(sample: dict) -> dict:
        async with sem:
            result = await council.predict(sample)
            pbar.update(1)
            return result

    results = await asyncio.gather(*[_process(s) for s in test_data])
    pbar.close()

    # --- Write predictions ---
    predictions = [{"id": r["id"], "label": r["label"]} for r in results]
    Path(args.output_path).write_text(json.dumps(predictions, indent=4, ensure_ascii=False))
    logger.info("Predictions saved to %s", args.output_path)

    if args.log_path:
        Path(args.log_path).write_text(json.dumps(list(results), indent=2, ensure_ascii=False))
        logger.info("Reasoning log saved to %s", args.log_path)

    # --- Distribution ---
    pred_labels = [r["label"] for r in results]
    print(format_distribution(pred_labels, title=f"Predictions [{args.model}]"))

    # --- Auto-evaluate if labels available ---
    if has_labels:
        y_true = [s["label"] for s in test_data]
        metrics = compute_metrics(y_true, pred_labels)
        print(format_report(metrics, title=f"Evaluation [{args.model}]"))
        metrics_path = Path(args.output_path).with_suffix(".metrics.json")
        metrics["model"] = args.model
        metrics["num_samples"] = len(y_true)
        Path(metrics_path).write_text(json.dumps(metrics, indent=2))
        logger.info("Metrics saved to %s", metrics_path)


# ---------------------------------------------------------------------------
# Evaluate (offline)
# ---------------------------------------------------------------------------

def run_evaluate(args: argparse.Namespace) -> None:
    from src.evaluate import evaluate_predictions
    evaluate_predictions(args.gold, args.pred, title=args.title)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "evaluate":
        run_evaluate(args)
    else:
        if not hasattr(args, "model"):
            parser.print_help()
            sys.exit(1)
        asyncio.run(run_predict(args))


if __name__ == "__main__":
    main()
