import json
from pathlib import Path

from scripts.sample_answers import (
    DEFAULT_IDS,
    DEFAULT_OUTPUT_PATH,
    EXIT_INVALID_ARGUMENT,
    EXIT_SUCCESS,
    answer_item,
    build_parser,
    main,
)
from src.agent.state import HopRecord
from src.baseline.schemas import PipelineResult
from src.eval.qa_dataset import QAItem
from src.generation.citations import Citation


def _item(question_id: str = "single_hop_01") -> QAItem:
    return QAItem(
        id=question_id,
        question="where is auth handled?",
        category="single_hop",
        expected_files=["auth.py"],
        grounding_notes="notes",
    )


def _fake_compare(question, cache_dir, embedder, reranker, backend):
    naive = PipelineResult(pipeline="naive", answer="naive says", citations=[], hop_path=[])
    ragnarok = PipelineResult(
        pipeline="ragnarok",
        answer="ragnarok says (auth.py#L1-L2)",
        citations=[Citation("auth.py", 1, 2, verified=True)],
        hop_path=[
            HopRecord(
                file_path="backend.py", symbol_name="Backend", start_line=3, end_line=9, reason="r"
            )
        ],
    )
    return naive, ragnarok


def _write_qa(path: Path, *question_ids: str) -> None:
    path.write_text("\n".join(_item(i).model_dump_json() for i in question_ids) + "\n")


class TestAnswerItem:
    def test_records_both_answers_with_citations_and_hops(self, monkeypatch, tmp_path):
        monkeypatch.setattr("scripts.sample_answers.compare", _fake_compare)

        record = answer_item(_item(), tmp_path, None, None, None)

        assert record["id"] == "single_hop_01"
        assert record["naive_answer"] == "naive says"
        assert record["ragnarok_answer"] == "ragnarok says (auth.py#L1-L2)"
        assert record["citations"] == [{"span": "auth.py#L1-L2", "verified": True}]
        assert record["hops"] == [{"symbol": "Backend", "file": "backend.py", "reason": "r"}]
        assert record["expected_files"] == ["auth.py"]


class TestBuildParser:
    def test_defaults(self):
        args = build_parser().parse_args(["/tmp/cache"])

        assert args.ids == list(DEFAULT_IDS)
        assert args.out == DEFAULT_OUTPUT_PATH
        assert args.device is None


class TestMain:
    def test_rejects_a_missing_index_directory(self, tmp_path):
        assert main([str(tmp_path / "missing")]) == EXIT_INVALID_ARGUMENT

    def test_rejects_an_unknown_question_id(self, tmp_path):
        qa_path = tmp_path / "qa.jsonl"
        _write_qa(qa_path, "single_hop_01")

        exit_code = main([str(tmp_path), "--qa-path", str(qa_path), "--ids", "nope"])

        assert exit_code == EXIT_INVALID_ARGUMENT

    def test_writes_each_answer_to_the_output_file(self, monkeypatch, tmp_path):
        qa_path, out_path = tmp_path / "qa.jsonl", tmp_path / "out.json"
        _write_qa(qa_path, "single_hop_01", "single_hop_02")
        monkeypatch.setattr("scripts.sample_answers.compare", _fake_compare)
        for model in ("Embedder", "Reranker", "CPUBackend"):
            monkeypatch.setattr(f"scripts.sample_answers.{model}", lambda *args, **kwargs: None)

        exit_code = main(
            [
                str(tmp_path),
                "--qa-path",
                str(qa_path),
                "--ids",
                "single_hop_02",
                "single_hop_01",
                "--out",
                str(out_path),
            ]
        )

        assert exit_code == EXIT_SUCCESS
        records = json.loads(out_path.read_text())
        assert [record["id"] for record in records] == ["single_hop_02", "single_hop_01"]
