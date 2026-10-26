
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import mean
from datetime import datetime, timezone

from evaluation.judge_utils import evaluate_sample, make_openrouter_judge

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "evaluation" / "gold_dataset.json"
RESULTS_DIR = ROOT_DIR / "evaluation" / "results"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> list[dict]:
    """Load and validate the evaluation dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Dataset must be a non-empty JSON list.")

    required_fields = ("question", "ground_truth", "context")

    for index, item in enumerate(dataset):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} must be a JSON object.")

        for field in required_fields:
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(
                    f"Item {index} has a missing or empty '{field}' field."
                )

    return dataset


def summarize_results(results: list[dict]) -> dict:
    """Calculate means from valid scores only."""
    metric_names = (
        "faithfulness",
        "answer_relevance",
        "context_precision",
    )

    summary = {}

    for metric in metric_names:
        scores = [
            item["metrics"][metric]["score"]
            for item in results
            if item["metrics"][metric]["score"] is not None
        ]

        summary[metric] = {
            "mean": round(mean(scores), 4) if scores else None,
            "successful": len(scores),
            "failed": len(results) - len(scores),
        }

    return summary


def run_evaluation(
    dataset: list[dict],
    judge_fn,
    limit: int | None = None,
) -> dict:
    """Evaluate supplied model outputs and retrieved contexts."""
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1.")
        dataset = dataset[:limit]

    results = []

    for index, item in enumerate(dataset, start=1):
        # The gold dataset does not contain a generated answer or
        # the actual context retrieved by the RAG system.
        # Do not treat the reference answer as a model answer.
        logger.warning(
            "Skipping item %d: generated answer and retrieved contexts "
            "are not available in the gold dataset.",
            index,
        )

        results.append({
            "index": index,
            "question": item["question"],
            "status": "skipped_missing_rag_output",
            "metrics": {
                "faithfulness": {"score": None, "error": "Missing generated answer and retrieved contexts."},
                "answer_relevance": {"score": None, "error": "Missing generated answer."},
                "context_precision": {"score": None, "error": "Missing retrieved contexts."},
            },
        })

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET_PATH.relative_to(ROOT_DIR)),
        "total_samples": len(results),
        "summary": summarize_results(results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the RAG evaluation pipeline."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of dataset samples to process.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the dataset without calling any API.",
    )
    args = parser.parse_args()

    dataset = load_dataset(DATASET_PATH)
    logger.info("Dataset validated: %d samples.", len(dataset))

    if args.validate_only:
        logger.info("Validation completed; no API calls were made.")
        return

    # Do not create a judge until real RAG outputs are available.
    report = run_evaluation(dataset, judge_fn=None, limit=args.limit)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "evaluation_results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    logger.info("Report saved to %s", output_path)
    logger.info(
        "No metrics were scored because generated answers and retrieved "
        "contexts are not available yet."
    )


if __name__ == "__main__":
    main()
