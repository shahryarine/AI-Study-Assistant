from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Callable, Sequence


logger = logging.getLogger(__name__)

JudgeFn = Callable[[str], str]


class MetricEvaluationError(Exception):
    def __init__(
        self,
        metric: str,
        reason: str,
        raw_response: str | None = None,
    ):
        self.metric = metric
        self.reason = reason
        self.raw_response = raw_response
        super().__init__(f"[{metric}] {reason}")




_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = _FENCE_RE.sub("", text.strip())

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end <= start:
            raise

        obj = json.loads(cleaned[start : end + 1])

    if not isinstance(obj, dict):
        raise ValueError("JSON root must be an object.")

    return obj


def _validate_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Score must be numeric: {value!r}")

    score = float(value)

    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"Score must be between 0 and 1: {value!r}")

    return score


def _call_judge(
    metric: str,
    judge_fn: JudgeFn,
    prompt: str,
    parser: Callable[[dict[str, Any]], Any],
    max_retries: int = 2,
) -> Any:
    last_reason = "Unknown error"
    raw_response = None

    for attempt in range(1, max_retries + 2):
        try:
            raw_response = judge_fn(prompt)
        except Exception as exc:
            last_reason = f"Judge request failed: {exc!r}"
            logger.error(
                "%s | attempt %d | %s",
                metric,
                attempt,
                last_reason,
            )
            continue

        try:
            return parser(_extract_json_object(raw_response))
        except (ValueError, KeyError, TypeError) as exc:
            last_reason = f"Invalid judge response: {exc}"
            logger.error(
                "%s | attempt %d | %s | response: %.500s",
                metric,
                attempt,
                last_reason,
                raw_response,
            )

    raise MetricEvaluationError(metric, last_reason, raw_response)


def _format_contexts(contexts: Sequence[str]) -> str:
    if not contexts:
        raise ValueError("Contexts cannot be empty.")

    if any(not isinstance(context, str) for context in contexts):
        raise ValueError("Every context must be a string.")

    return "\n\n".join(
        f"[{index}] {context}"
        for index, context in enumerate(contexts, start=1)
    )


FAITHFULNESS_PROMPT = """You are a strict evaluator of factual grounding.

Break the ANSWER into atomic factual claims. For each claim, determine whether
it can be inferred directly from the CONTEXT alone. Do not use outside knowledge.
Claims not supported by the context must be marked unsupported.

CONTEXT:
{context}

ANSWER:
{answer}

Return only a valid JSON object in this format:
{{"claims": [{{"claim": "text", "supported": true}}]}}
"""


def _parse_faithfulness(obj: dict[str, Any]) -> float:
    claims = obj["claims"]

    if not isinstance(claims, list) or not claims:
        raise ValueError("Claims must be a non-empty list.")

    supported_count = 0

    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("Each claim must be an object.")

        supported = claim.get("supported")

        if not isinstance(supported, bool):
            raise ValueError("The supported field must be a boolean.")

        supported_count += int(supported)

    return supported_count / len(claims)


def evaluate_faithfulness(
    answer: str,
    contexts: Sequence[str],
    judge_fn: JudgeFn,
    max_retries: int = 2,
) -> float:
    prompt = FAITHFULNESS_PROMPT.format(
        context=_format_contexts(contexts),
        answer=answer,
    )

    return _call_judge(
        "faithfulness",
        judge_fn,
        prompt,
        _parse_faithfulness,
        max_retries,
    )


ANSWER_RELEVANCE_PROMPT = """You are an evaluator of answer relevance.

Judge how directly and completely the ANSWER addresses the QUESTION.
Ignore factual correctness. Penalize missing information, irrelevant content,
evasion, and unnecessary padding.

Scoring rubric:
1.0 = fully and directly answers every part
0.7 = answers the question with minor gaps
0.4 = partially answers or is mostly tangential
0.0 = does not answer the question

QUESTION:
{question}

ANSWER:
{answer}

Return only a valid JSON object in this format:
{{"score": 0.0}}
"""


def _parse_score_object(obj: dict[str, Any]) -> float:
    return _validate_score(obj["score"])


def evaluate_answer_relevance(
    question: str,
    answer: str,
    judge_fn: JudgeFn,
    max_retries: int = 2,
) -> float:
    prompt = ANSWER_RELEVANCE_PROMPT.format(
        question=question,
        answer=answer,
    )

    return _call_judge(
        "answer_relevance",
        judge_fn,
        prompt,
        _parse_score_object,
        max_retries,
    )


CONTEXT_PRECISION_PROMPT = """You are an evaluator of retrieval quality.

For each retrieved chunk, decide whether it contains information useful
for answering the QUESTION and supporting the REFERENCE ANSWER.
Evaluate each chunk independently. Do not reorder the chunks.

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

RETRIEVED CHUNKS IN RETRIEVAL ORDER:
{context}

Return only a valid JSON object with one verdict per chunk:
{{"verdicts": [{{"chunk": 1, "relevant": true}}]}}
"""


def _make_precision_parser(
    n_chunks: int,
) -> Callable[[dict[str, Any]], float]:
    def parse(obj: dict[str, Any]) -> float:
        verdicts = obj["verdicts"]

        if not isinstance(verdicts, list) or len(verdicts) != n_chunks:
            raise ValueError(
                f"Expected exactly {n_chunks} verdicts."
            )

        relevance: dict[int, bool] = {}

        for verdict in verdicts:
            if not isinstance(verdict, dict):
                raise ValueError("Each verdict must be an object.")

            chunk = verdict.get("chunk")
            relevant = verdict.get("relevant")

            if (
                isinstance(chunk, bool)
                or not isinstance(chunk, int)
                or not 1 <= chunk <= n_chunks
            ):
                raise ValueError(f"Invalid chunk number: {chunk!r}")

            if not isinstance(relevant, bool):
                raise ValueError("The relevant field must be a boolean.")

            if chunk in relevance:
                raise ValueError(f"Duplicate chunk number: {chunk}")

            relevance[chunk] = relevant

        if set(relevance) != set(range(1, n_chunks + 1)):
            raise ValueError("One or more chunk verdicts are missing.")

        hits = 0
        precision_sum = 0.0

        for rank in range(1, n_chunks + 1):
            if relevance[rank]:
                hits += 1
                precision_sum += hits / rank

        return precision_sum / hits if hits else 0.0

    return parse


def evaluate_context_precision(
    question: str,
    contexts: Sequence[str],
    reference_answer: str,
    judge_fn: JudgeFn,
    max_retries: int = 2,
) -> float:
    if not contexts:
        raise MetricEvaluationError(
            "context_precision",
            "Retrieved contexts cannot be empty.",
        )

    prompt = CONTEXT_PRECISION_PROMPT.format(
        question=question,
        reference=reference_answer,
        context=_format_contexts(contexts),
    )

    parser = _make_precision_parser(len(contexts))

    return _call_judge(
        "context_precision",
        judge_fn,
        prompt,
        parser,
        max_retries,
    )