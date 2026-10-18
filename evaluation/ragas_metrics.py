
import json
import os
from pathlib import Path
from statistics import mean

from rag.generator import OpenRouterClient


class RagasEvaluator:
    """LLM-as-a-Judge evaluator for a RAG dataset."""

    def __init__(self, dataset_path=None, output_path=None):
        project_root = Path(__file__).resolve().parent.parent

        self.dataset_path = Path(
            dataset_path or project_root / "evaluation" / "gold_dataset.json"
        )
        self.output_path = Path(
            output_path or project_root / "evaluation" / "results.json"
        )

        self.llm_client = OpenRouterClient()

    def load_dataset(self):
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        with self.dataset_path.open("r", encoding="utf-8") as file:
            dataset = json.load(file)

        if not isinstance(dataset, list):
            raise ValueError("The dataset must be a JSON array.")

        required_fields = {"question", "ground_truth"}
        for index, item in enumerate(dataset, start=1):
            missing = required_fields - item.keys()
            if missing:
                raise ValueError(
                    f"Item {index} is missing fields: {sorted(missing)}"
                )

        return dataset

    def judge_score(self, metric, question, answer, contexts,
                    ground_truth=""):
        context_text = "\n\n".join(contexts)

        prompts = {
            "faithfulness": (
                "Evaluate whether every factual claim in the answer "
                "is supported by the retrieved contexts. "
                "Ignore claims that are clearly marked as uncertainty."
            ),
            "answer_relevance": (
                "Evaluate how directly and completely the answer "
                "addresses the question. Do not reward irrelevant content."
            ),
            "context_precision": (
                "Evaluate how relevant the retrieved contexts are "
                "to answering the question. Give a high score when "
                "the contexts are useful and focused, and a low score "
                "when they are mostly irrelevant."
            ),
        }

        prompt = f"""
You are a strict and impartial RAG evaluation judge.

Metric: {metric}
Criterion: {prompts[metric]}

Question:
{question}

Reference answer (for guidance only):
{ground_truth}

Retrieved contexts:
{context_text}

Generated answer:
{answer}

Return only a JSON object in this exact format:
{{"score": 0.0}}

The score must be a number between 0.0 and 1.0.
Do not include any text outside the JSON object.
"""

        response = self.llm_client.generate(
            system_prompt=(
                "You are an impartial evaluation judge. "
                "Return valid JSON only."
            ),
            user_prompt=prompt,
            is_json_mode=True,
        )

        if isinstance(response, dict):
            result = response
        else:
            result = json.loads(response)

        score = float(result["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Invalid score: {score}")

        return score

    def run_evaluation(self):
        dataset = self.load_dataset()
        results = []

        print(f"Loaded {len(dataset)} questions.")

        for index, item in enumerate(dataset, start=1):
            question = item["question"]
            ground_truth = item["ground_truth"]
            answer = item.get("answer")

            contexts = item.get("retrieved_contexts")
            if contexts is None:
                fallback_context = item.get("context", "")
                contexts = (
                    [fallback_context]
                    if isinstance(fallback_context, str)
                    else fallback_context
                )

            if not answer:
                print(
                    f"Skipping item {index}: "
                    "no generated 'answer' is available."
                )
                continue

            if isinstance(contexts, str):
                contexts = [contexts]

            if not contexts:
                print(
                    f"Skipping item {index}: "
                    "no retrieved contexts are available."
                )
                continue

            print(f"Evaluating {index}/{len(dataset)}...")

            item_result = {
                "question": question,
                "faithfulness": None,
                "answer_relevance": None,
                "context_precision": None,
            }

            for metric in (
                "faithfulness",
                "answer_relevance",
                "context_precision",
            ):
                try:
                    item_result[metric] = self.judge_score(
                        metric=metric,
                        question=question,
                        answer=answer,
                        contexts=contexts,
                        ground_truth=ground_truth,
                    )
                except Exception as error:
                    print(f"  {metric} failed: {error}")

            results.append(item_result)

        valid_results = [
            result for result in results
            if any(
                result[metric] is not None
                for metric in (
                    "faithfulness",
                    "answer_relevance",
                    "context_precision",
                )
            )
        ]

        summary = {}
        for metric in (
            "faithfulness",
            "answer_relevance",
            "context_precision",
        ):
            scores = [
                result[metric]
                for result in valid_results
                if result[metric] is not None
            ]
            summary[metric] = {
                "mean": round(mean(scores), 4) if scores else None,
                "count": len(scores),
            }

        output = {
            "evaluation_method": "custom_llm_as_a_judge",
            "dataset_path": str(self.dataset_path),
            "total_questions": len(dataset),
            "evaluated_questions": len(valid_results),
            "metrics_summary": summary,
            "results": results,
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as file:
            json.dump(output, file, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: {self.output_path}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    evaluator = RagasEvaluator()
    evaluator.run_evaluation()
