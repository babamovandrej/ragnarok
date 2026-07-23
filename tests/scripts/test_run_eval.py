import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_eval import EXIT_INVALID_ARGUMENT, EXIT_SUCCESS, build_parser, main
from src.baseline.naive_rag import PipelineResult
from src.config import DEFAULT_RETRIEVAL_TOP_K
from src.eval.ablation import AblationRow


@pytest.fixture
def qa_path(tmp_path) -> Path:
    path = tmp_path / "qa.jsonl"
    item = {
        "id": "x",
        "question": "how is auth handled?",
        "category": "single_hop",
        "expected_files": ["a.py"],
        "grounding_notes": "n",
        "expect_idk": False,
    }
    path.write_text(json.dumps(item) + "\n")
    return path


@pytest.fixture
def cache_dir(tmp_path) -> Path:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    return index_dir


class _NoopModel:
    pass


class _FakeScored:
    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "faithfulness": [0.8],
                "llm_context_precision_without_reference": [0.7],
                "context_recall": [0.6],
            }
        )


class TestBuildParser:
    def test_parses_cache_dir(self):
        args = build_parser().parse_args(["/tmp/cache"])

        assert args.cache_dir == Path("/tmp/cache")

    def test_defaults_come_from_config(self):
        args = build_parser().parse_args(["/tmp/cache"])

        assert args.top_k == DEFAULT_RETRIEVAL_TOP_K
        assert args.ablation is False

    def test_ablation_flag_can_be_set(self):
        args = build_parser().parse_args(["/tmp/cache", "--ablation"])

        assert args.ablation is True


class TestMain:
    def test_missing_index_returns_invalid_argument(self, tmp_path, qa_path):
        exit_code = main([str(tmp_path / "missing"), "--qa-path", str(qa_path)])

        assert exit_code == EXIT_INVALID_ARGUMENT

    def test_missing_qa_path_returns_invalid_argument(self, cache_dir, tmp_path):
        exit_code = main([str(cache_dir), "--qa-path", str(tmp_path / "missing.jsonl")])

        assert exit_code == EXIT_INVALID_ARGUMENT

    def test_default_mode_prints_ragas_scores_and_idk_rate(
        self, cache_dir, qa_path, monkeypatch, capsys
    ):
        monkeypatch.setattr("scripts.run_eval.Embedder", _NoopModel)
        monkeypatch.setattr("scripts.run_eval.Reranker", _NoopModel)
        monkeypatch.setattr("scripts.run_eval.CPUBackend", _NoopModel)
        monkeypatch.setattr(
            "scripts.run_eval.answer_ragnarok",
            lambda *a, **k: PipelineResult(
                pipeline="ragnarok",
                answer="I don't know.",
                citations=[],
                hop_path=[],
                context_texts=["ctx"],
            ),
        )
        monkeypatch.setattr("scripts.run_eval.run_ragas_eval", lambda *a, **k: _FakeScored())

        exit_code = main([str(cache_dir), "--qa-path", str(qa_path)])

        assert exit_code == EXIT_SUCCESS
        out = capsys.readouterr().out
        assert "Aggregate faithfulness" in out
        assert "Adversarial IDK rate" in out

    def test_ablation_mode_prints_a_three_row_table(self, cache_dir, qa_path, monkeypatch, capsys):
        monkeypatch.setattr("scripts.run_eval.Embedder", _NoopModel)
        monkeypatch.setattr("scripts.run_eval.Reranker", _NoopModel)
        monkeypatch.setattr("scripts.run_eval.CPUBackend", _NoopModel)
        monkeypatch.setattr(
            "scripts.run_eval.run_ablation",
            lambda *a, **k: [
                AblationRow("naive", 0.5, 0.5, 0.5),
                AblationRow("hybrid_rerank", 0.7, 0.7, 0.7),
                AblationRow("multi_hop", 0.9, 0.9, 0.9),
            ],
        )

        exit_code = main([str(cache_dir), "--qa-path", str(qa_path), "--ablation"])

        assert exit_code == EXIT_SUCCESS
        out = capsys.readouterr().out
        assert "naive" in out
        assert "hybrid_rerank" in out
        assert "multi_hop" in out
