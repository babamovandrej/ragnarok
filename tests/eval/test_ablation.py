from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.baseline.schemas import PipelineResult
from src.eval.ablation import _run_pipeline, run_ablation
from src.eval.qa_dataset import QAItem


def _qa_item(question: str = "q") -> QAItem:
    return QAItem(
        id="x",
        question=question,
        category="single_hop",
        expected_files=["a.py"],
        grounding_notes="n",
    )


def _pipeline_result(pipeline: str, answer: str = "an answer") -> PipelineResult:
    return PipelineResult(
        pipeline=pipeline, answer=answer, citations=[], hop_path=[], context_texts=["ctx"]
    )


class _FakeScored:
    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "faithfulness": [0.8],
                "llm_context_precision_without_reference": [0.7],
                "context_recall": [0.6],
            }
        )


class TestRunPipeline:
    def test_naive_pipeline_calls_answer_naive(self):
        embedder, reranker, backend = object(), object(), object()
        with patch("src.eval.ablation.answer_naive") as mock_naive:
            mock_naive.return_value = _pipeline_result("naive")

            result = _run_pipeline(
                "naive", "q", Path("/cache"), embedder, reranker, backend, top_k=5
            )

            mock_naive.assert_called_once_with("q", Path("/cache"), embedder, backend, top_k=5)
            assert result.pipeline == "naive"

    def test_hybrid_rerank_pipeline_calls_answer_ragnarok_with_zero_hops(self):
        embedder, reranker, backend = object(), object(), object()
        with patch("src.eval.ablation.answer_ragnarok") as mock_ragnarok:
            mock_ragnarok.return_value = _pipeline_result("ragnarok")

            _run_pipeline(
                "hybrid_rerank", "q", Path("/cache"), embedder, reranker, backend, top_k=5
            )

            mock_ragnarok.assert_called_once_with(
                "q", Path("/cache"), embedder, reranker, backend, top_k=5, max_hops=0
            )

    def test_multi_hop_pipeline_calls_answer_ragnarok_with_default_hops(self):
        embedder, reranker, backend = object(), object(), object()
        with patch("src.eval.ablation.answer_ragnarok") as mock_ragnarok:
            mock_ragnarok.return_value = _pipeline_result("ragnarok")

            _run_pipeline("multi_hop", "q", Path("/cache"), embedder, reranker, backend, top_k=5)

            mock_ragnarok.assert_called_once_with(
                "q", Path("/cache"), embedder, reranker, backend, top_k=5
            )


class TestRunAblation:
    def test_produces_one_row_per_pipeline_with_scores_from_ragas(self):
        qa_items = [_qa_item("q1"), _qa_item("q2")]
        embedder, reranker, backend = object(), object(), object()

        with (
            patch("src.eval.ablation.answer_naive", return_value=_pipeline_result("naive")),
            patch("src.eval.ablation.answer_ragnarok", return_value=_pipeline_result("ragnarok")),
            patch("src.eval.ablation.run_ragas_eval", return_value=_FakeScored()) as mock_eval,
        ):
            rows = run_ablation(qa_items, Path("/cache"), embedder, reranker, backend, top_k=5)

        assert [row.pipeline for row in rows] == ["naive", "hybrid_rerank", "multi_hop"]
        for row in rows:
            assert row.faithfulness == pytest.approx(0.8)
            assert row.context_precision == pytest.approx(0.7)
            assert row.context_recall == pytest.approx(0.6)
        assert mock_eval.call_count == 3
