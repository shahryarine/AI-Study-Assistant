
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from evaluation.ragas_metrics import (
    JudgeFn,
    MetricEvaluationError,
    evaluate_answer_relevance,
    evaluate_context_precision,
    evaluate_faithfulness,
)

if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def make_openrouter_judge(
    model: str | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
    network_retries: int = 3,
) -> JudgeFn:
    key = api_key or os.environ.get("OPENROUTER_API_KEY")

    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set in the environment or .env file."
        )

    model_name = model or os.environ.get(
        "OPENROUTER_MODEL", DEFAULT_MODEL
    )

    if timeout <= 0:
        raise ValueError("timeout must be greater than zero.")

    if network_retries < 0:
        raise ValueError("network_retries cannot be negative.")

    def judge(prompt: str) -> str:
        body = json.dumps(
            {
                "model": model_name,
                "temperature": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a strict evaluation judge. "
                            "Follow the user's requested output format exactly."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            OPENROUTER_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error: Exception | None = None

        for attempt in range(network_retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=timeout
                ) as response:
                    payload = json.load(response)

                content = payload["choices"][0]["message"]["content"]

                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError(
                        "OpenRouter returned empty or invalid content."
                    )

                return content.strip()

            except urllib.error.HTTPError as exc:
                last_error = exc

                if exc.code not in (429, 500, 502, 503, 504):
                    break

            except (
                urllib.error.URLError,
                TimeoutError,
                KeyError,
                IndexError,
                TypeError,
                json.JSONDecodeError,
                RuntimeError,
            ) as exc:
                last_error = exc

            if attempt < network_retries:
                time.sleep(2 ** attempt)

        if last_error is None:
            raise RuntimeError("OpenRouter request failed for an unknown reason.")

        if isinstance(last_error, urllib.error.HTTPError):
            detail = f"HTTP {last_error.code}"
        else:
            detail = type(last_error).__name__

        raise RuntimeError(f"OpenRouter request failed: {detail}") from None

    return judge


def evaluate_sample(
    question: str,
    answer: str,
    contexts: Sequence[str],
    reference_answer: str,
    judge_fn: JudgeFn,
) -> dict[str, dict[str, Any]]:
    metrics = {
        "faithfulness": lambda: evaluate_faithfulness(
            answer, contexts, judge_fn
        ),
        "answer_relevance": lambda: evaluate_answer_relevance(
            question, answer, judge_fn
        ),
        "context_precision": lambda: evaluate_context_precision(
            question, contexts, reference_answer, judge_fn
        ),
    }

    results: dict[str, dict[str, Any]] = {}

    for name, metric_function in metrics.items():
        try:
            results[name] = {
                "score": metric_function(),
                "error": None,
            }

        except MetricEvaluationError as exc:
            results[name] = {
                "score": None,
                "error": exc.reason,
            }

        except Exception as exc:
            logger.exception("%s evaluation failed unexpectedly.", name)
            results[name] = {
                "score": None,
                "error": f"Unexpected error: {type(exc).__name__}",
            }

    return results
