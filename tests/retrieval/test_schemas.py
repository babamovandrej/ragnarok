import dataclasses

import pytest

from src.retrieval.schemas import RetrievedChunk, chunk_from_chroma_record


class TestChunkFromChromaRecord:
    def test_reassembles_all_chunk_fields(self):
        metadata = {
            "file_path": "pkg/mod.py",
            "start_line": 10,
            "end_line": 20,
            "symbol_name": "foo",
            "qualified_name": "Example.foo",
            "symbol_type": "method",
            "language": "python",
        }

        chunk = chunk_from_chroma_record("chunk-1", "def foo():\n    pass", metadata)

        assert chunk.chunk_id == "chunk-1"
        assert chunk.file_path == "pkg/mod.py"
        assert chunk.start_line == 10
        assert chunk.end_line == 20
        assert chunk.symbol_name == "foo"
        assert chunk.qualified_name == "Example.foo"
        assert chunk.symbol_type == "method"
        assert chunk.language == "python"
        assert chunk.text == "def foo():\n    pass"


class TestRetrievedChunk:
    def test_is_frozen(self):
        retrieved = RetrievedChunk(
            chunk=chunk_from_chroma_record(
                "chunk-1",
                "pass",
                {
                    "file_path": "a.py",
                    "start_line": 1,
                    "end_line": 1,
                    "symbol_name": "foo",
                    "qualified_name": "foo",
                    "symbol_type": "function",
                    "language": "python",
                },
            ),
            score=0.5,
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(retrieved, "score", 1.0)
