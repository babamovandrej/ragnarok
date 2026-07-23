from pathlib import Path

import pytest
from pydantic import ValidationError

from src.eval.qa_dataset import QAItem, load_qa_dataset

QA_PATH = Path(__file__).parent.parent.parent / "data" / "eval" / "qa_starlette.jsonl"


class TestQAItem:
    def test_defaults_expect_idk_to_false(self):
        item = QAItem(
            id="x",
            question="q",
            category="single_hop",
            expected_files=["a.py"],
            grounding_notes="n",
        )

        assert item.expect_idk is False

    def test_rejects_an_unknown_category(self):
        with pytest.raises(ValidationError):
            QAItem(
                id="x",
                question="q",
                category="no_such_category",
                expected_files=[],
                grounding_notes="n",
            )


class TestLoadQADataset:
    def test_loads_between_30_and_50_well_formed_items(self):
        items = load_qa_dataset(QA_PATH)

        assert 30 <= len(items) <= 50
        assert all(isinstance(item, QAItem) for item in items)

    def test_ids_are_unique(self):
        items = load_qa_dataset(QA_PATH)

        assert len(items) == len({item.id for item in items})

    def test_covers_all_three_categories(self):
        items = load_qa_dataset(QA_PATH)

        assert {item.category for item in items} == {"single_hop", "multi_hop", "adversarial"}

    def test_single_hop_and_multi_hop_items_have_expected_files(self):
        items = load_qa_dataset(QA_PATH)

        for item in items:
            if item.category in ("single_hop", "multi_hop"):
                assert item.expected_files, item.id

    def test_multi_hop_items_expect_at_least_two_files(self):
        items = load_qa_dataset(QA_PATH)

        for item in items:
            if item.category == "multi_hop":
                assert len(item.expected_files) >= 2, item.id

    def test_adversarial_items_expect_idk_and_no_expected_files(self):
        items = load_qa_dataset(QA_PATH)

        for item in items:
            if item.category == "adversarial":
                assert item.expect_idk is True, item.id
                assert item.expected_files == [], item.id

    def test_adversarial_items_are_at_least_four(self):
        items = load_qa_dataset(QA_PATH)

        adversarial = [item for item in items if item.category == "adversarial"]
        assert len(adversarial) >= 4
