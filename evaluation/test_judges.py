from __future__ import annotations

import json
import os
import urllib.error

from evaluation.judge_utils import evaluate_sample, make_openrouter_judge
from evaluation.ragas_metrics import (
    MetricEvaluationError,
    evaluate_answer_relevance,
    evaluate_context_precision,
    evaluate_faithfulness,
)

QUESTION = "پایتخت فرانسه کجاست؟"
ANSWER = "پایتخت فرانسه پاریس است."
CONTEXTS = ["پاریس پایتخت و بزرگ‌ترین شهر فرانسه است.", "برج ایفل در پاریس قرار دارد."]
REFERENCE = "پاریس"

GOOD = {
    "faithfulness": '{"claims": [{"claim": "x", "supported": true, "reason": "r"}]}',
    "answer_relevance": '{"score": 0.9, "reason": "r"}',
    "context_precision": json.dumps(
        {"verdicts": [{"chunk": 1, "relevant": True}, {"chunk": 2, "relevant": False}]}
    ),
}


def fake_judge(responses: dict[str, str | Exception]):

    def judge(prompt: str) -> str:
        if "factual grounding" in prompt:
            r = responses["faithfulness"]
        elif "answer relevance" in prompt:
            r = responses["answer_relevance"]
        else:
            r = responses["context_precision"]
        if isinstance(r, Exception):
            raise r
        return r

    return judge


def _assert_valid(score):
    assert isinstance(score, float) and 0.0 <= score <= 1.0, score


def test_each_metric_returns_valid_score():
    j = fake_judge(GOOD)
    _assert_valid(evaluate_faithfulness(ANSWER, CONTEXTS, j))
    _assert_valid(evaluate_answer_relevance(QUESTION, ANSWER, j))
    _assert_valid(evaluate_context_precision(QUESTION, CONTEXTS, REFERENCE, j))


def _expect_error(fn):
    try:
        fn()
    except MetricEvaluationError:
        return
    raise AssertionError("باید MetricEvaluationError پرتاب می‌شد")


def test_invalid_json_raises_not_zero():
    j = fake_judge({**GOOD, "answer_relevance": "این JSON نیست"})
    _expect_error(lambda: evaluate_answer_relevance(QUESTION, ANSWER, j, max_retries=1))


def test_non_numeric_and_out_of_range_scores_raise():
    for bad in ['{"score": "0.9"}', '{"score": 1.5}', '{"score": -0.1}', '{"score": true}', '{"score": null}', '{}']:
        j = fake_judge({**GOOD, "answer_relevance": bad})
        _expect_error(lambda: evaluate_answer_relevance(QUESTION, ANSWER, j, max_retries=0))


def test_network_error_raises_metric_error():
    err = urllib.error.URLError("connection refused")
    j = fake_judge({**GOOD, "faithfulness": err})
    _expect_error(lambda: evaluate_faithfulness(ANSWER, CONTEXTS, j, max_retries=1))


def test_retry_recovers_after_transient_failure():
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("timeout")
        return GOOD["answer_relevance"]

    assert evaluate_answer_relevance(QUESTION, ANSWER, flaky, max_retries=2) == 0.9


def test_one_metric_failure_keeps_the_others():
    for failing, bad in [
        ("faithfulness", urllib.error.URLError("down")),
        ("answer_relevance", "not json"),
        ("context_precision", '{"verdicts": [{"chunk": 1, "relevant": "yes"}]}'),
    ]:
        res = evaluate_sample(QUESTION, ANSWER, CONTEXTS, REFERENCE, fake_judge({**GOOD, failing: bad}))
        assert res[failing]["score"] is None and res[failing]["error"], res
        for name, r in res.items():
            if name != failing:
                assert r["error"] is None
                _assert_valid(r["score"])


def test_context_precision_edge_cases():
    none_rel = json.dumps({"verdicts": [{"chunk": 1, "relevant": False}, {"chunk": 2, "relevant": False}]})
    j = fake_judge({**GOOD, "context_precision": none_rel})
    assert evaluate_context_precision(QUESTION, CONTEXTS, REFERENCE, j) == 0.0  # معتبر، نه خطا
    _expect_error(lambda: evaluate_context_precision(QUESTION, [], REFERENCE, j))
    wrong_count = json.dumps({"verdicts": [{"chunk": 1, "relevant": True}]})
    j = fake_judge({**GOOD, "context_precision": wrong_count})
    _expect_error(lambda: evaluate_context_precision(QUESTION, CONTEXTS, REFERENCE, j, max_retries=0))


def live_test():
    judge = make_openrouter_judge()
    res = evaluate_sample(QUESTION, ANSWER, CONTEXTS, REFERENCE, judge)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    for name, r in res.items():
        assert r["error"] is None, f"{name}: {r['error']}"
        _assert_valid(r["score"])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("OK  ", t.__name__)
    if os.environ.get("OPENROUTER_API_KEY"):
        live_test()
        print("OK   live_test")
    else:
        print("SKIP live_test (OPENROUTER_API_KEY تنظیم نشده)")
