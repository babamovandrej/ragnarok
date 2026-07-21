from src.ingestion.schemas import Chunk
from src.retrieval.hybrid import fuse_rrf
from src.retrieval.schemas import RetrievedChunk


def _fake_chunk(name: str) -> Chunk:
    return Chunk(
        chunk_id=name,
        file_path="a.py",
        start_line=1,
        end_line=2,
        symbol_name=name,
        qualified_name=name,
        symbol_type="function",
        language="python",
        text=f"def {name}():\n    pass",
    )


def _ranked(*names: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk=_fake_chunk(name), score=1.0 / (index + 1))
        for index, name in enumerate(names)
    ]


class TestFuseRrf:
    def test_fused_ranking_differs_from_either_input_ranking(self):
        dense = _ranked("a", "b", "c")
        sparse = _ranked("c", "b", "a")

        fused = fuse_rrf(dense, sparse, top_k=3)

        fused_order = [r.chunk.chunk_id for r in fused]
        assert fused_order != [r.chunk.chunk_id for r in dense]
        assert fused_order != [r.chunk.chunk_id for r in sparse]
        assert fused_order == ["a", "c", "b"]

    def test_dedupes_by_chunk_id(self):
        dense = _ranked("a")
        sparse = _ranked("a")

        fused = fuse_rrf(dense, sparse, top_k=10)

        assert len(fused) == 1
        assert fused[0].chunk.chunk_id == "a"

    def test_respects_top_k(self):
        dense = _ranked("a", "b", "c")

        fused = fuse_rrf(dense, top_k=2)

        assert len(fused) == 2

    def test_single_list_preserves_its_own_order(self):
        dense = _ranked("a", "b")

        fused = fuse_rrf(dense, top_k=2)

        assert [r.chunk.chunk_id for r in fused] == ["a", "b"]

    def test_chunk_appearing_in_both_lists_outranks_a_single_appearance(self):
        dense = _ranked("a", "b")
        sparse = _ranked("b", "c")

        fused = fuse_rrf(dense, sparse, top_k=3)

        assert fused[0].chunk.chunk_id == "b"
