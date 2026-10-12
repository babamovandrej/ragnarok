from pathlib import Path

import numpy as np
import pytest

from src.agent.tools import (
    RelevantReferences,
    find_symbol_definition,
    referenced_symbols,
    retrieve_hybrid,
)
from src.ingestion.indexer import BM25_DIRNAME, build_index, load_chroma_collection
from src.ingestion.schemas import Chunk
from src.retrieval.schemas import RetrievedChunk


def _fake_chunk(name: str, text: str, file_path: str = "a.py") -> Chunk:
    return Chunk(
        chunk_id=name,
        file_path=file_path,
        start_line=1,
        end_line=2,
        symbol_name=name,
        qualified_name=name,
        symbol_type="function",
        language="python",
        repo_sha="sha1",
        text=text,
    )


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 4), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]


class _FakeReranker:
    def rerank(self, query, candidates):
        return candidates


@pytest.fixture
def cache_dir(tmp_path) -> Path:
    chunks = [
        _fake_chunk(
            "AuthenticationMiddleware",
            "class AuthenticationMiddleware:\n    def __call__(self, request):\n        pass",
            "starlette/authentication.py",
        ),
        _fake_chunk("add", "def add(a, b):\n    return a + b"),
    ]
    embeddings = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    build_index(chunks, embeddings, tmp_path, repo_slug="a/b", repo_sha="sha1")
    return tmp_path


class TestFindSymbolDefinition:
    def test_resolves_a_known_symbol_to_its_exact_defining_chunk(self, cache_dir):
        collection = load_chroma_collection(cache_dir)

        found = find_symbol_definition("AuthenticationMiddleware", collection)

        assert found is not None
        assert found.chunk.file_path == "starlette/authentication.py"
        assert found.chunk.symbol_name == "AuthenticationMiddleware"

    def test_returns_none_for_an_unknown_symbol(self, cache_dir):
        collection = load_chroma_collection(cache_dir)

        assert find_symbol_definition("NoSuchSymbol", collection) is None


class TestRetrieveHybrid:
    def test_returns_reranked_candidates_from_dense_and_sparse_fusion(self, cache_dir):
        collection = load_chroma_collection(cache_dir)

        results = retrieve_hybrid(
            "auth", collection, cache_dir / BM25_DIRNAME, _FakeEmbedder(), _FakeReranker(), top_k=2
        )

        assert {r.chunk.symbol_name for r in results} == {"AuthenticationMiddleware", "add"}


class _RaisingCollection:
    """A chromadb.Collection stand-in that always fails, for tool-call failure-path tests.

    tools.py deliberately has no try/except of its own — graph.py's retrieve_node/
    follow_symbol_node are what catch and degrade — so these confirm the
    exception actually propagates up to that boundary instead of being silently swallowed
    somewhere in between.
    """

    def get(self, *args, **kwargs):
        raise RuntimeError("chroma unavailable")

    def query(self, *args, **kwargs):
        raise RuntimeError("chroma unavailable")


class TestFindSymbolDefinitionErrorHandling:
    def test_propagates_a_collection_failure_instead_of_swallowing_it(self):
        with pytest.raises(RuntimeError, match="chroma unavailable"):
            find_symbol_definition("AnySymbol", _RaisingCollection())


class TestRetrieveHybridErrorHandling:
    def test_propagates_a_collection_failure_instead_of_swallowing_it(self, tmp_path):
        with pytest.raises(RuntimeError, match="chroma unavailable"):
            retrieve_hybrid(
                "query",
                _RaisingCollection(),
                tmp_path,
                _FakeEmbedder(),
                _FakeReranker(),
                top_k=2,
            )


def _definition(
    name: str,
    text: str,
    *,
    symbol_type: str = "function",
    file_path: str = "pkg.py",
    start_line: int = 1,
    end_line: int | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=f"{file_path}:{name}:{start_line}",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line if end_line is not None else start_line + 1,
        symbol_name=name,
        qualified_name=name,
        symbol_type=symbol_type,
        language="python",
        repo_sha="sha1",
        text=text,
    )


def _collection_of(tmp_path: Path, chunks: list[Chunk]):
    embeddings = np.eye(len(chunks), 4, dtype=np.float32)
    build_index(chunks, embeddings, tmp_path, repo_slug="a/b", repo_sha="sha1")
    return load_chroma_collection(tmp_path)


def _retrieved(chunk: Chunk) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, score=1.0)


class TestReferencedSymbols:
    def test_lists_definitions_the_context_refers_to_but_does_not_contain(self, tmp_path):
        requires = _definition("requires", "def requires():\n    return RedirectResponse(url)")
        redirect = _definition("RedirectResponse", "class RedirectResponse: ...", start_line=10)
        unrelated = _definition("add", "def add(a, b): ...", start_line=20)
        collection = _collection_of(tmp_path, [requires, redirect, unrelated])

        assert referenced_symbols([_retrieved(requires)], collection) == ["RedirectResponse"]

    def test_leaves_out_definitions_already_in_context(self, tmp_path):
        requires = _definition("requires", "def requires():\n    return RedirectResponse(url)")
        redirect = _definition("RedirectResponse", "class RedirectResponse: ...", start_line=10)
        collection = _collection_of(tmp_path, [requires, redirect])

        context = [_retrieved(requires), _retrieved(redirect)]

        assert referenced_symbols(context, collection) == []

    def test_leaves_out_definitions_nested_inside_a_context_chunk(self, tmp_path):
        outer = _definition(
            "requires", "def requires():\n    def wrapper(): ...\n    return wrapper", end_line=9
        )
        nested = _definition("wrapper", "def wrapper(): ...", start_line=2, end_line=3)
        collection = _collection_of(tmp_path, [outer, nested])

        assert referenced_symbols([_retrieved(outer)], collection) == []

    def test_leaves_out_names_the_repo_defines_more_than_once(self, tmp_path):
        caller = _definition("caller", "def caller():\n    return close()")
        first = _definition("close", "def close(): ...", file_path="a.py")
        second = _definition("close", "def close(): ...", file_path="b.py")
        collection = _collection_of(tmp_path, [caller, first, second])

        assert referenced_symbols([_retrieved(caller)], collection) == []

    def test_orders_types_before_functions_before_methods_within_a_chunk(self, tmp_path):
        caller = _definition("caller", "def caller():\n    url; helper(); Response")
        method = _definition("url", "def url(self): ...", symbol_type="method", start_line=10)
        function = _definition("helper", "def helper(): ...", start_line=20)
        klass = _definition("Response", "class Response: ...", symbol_type="class", start_line=30)
        collection = _collection_of(tmp_path, [caller, method, function, klass])

        names = referenced_symbols([_retrieved(caller)], collection)

        assert names == ["Response", "helper", "url"]

    def test_earlier_context_chunks_contribute_first_and_the_list_is_capped(self, tmp_path):
        first = _definition("first", "def first():\n    return Beta()")
        second = _definition("second", "def second():\n    return Alpha()", start_line=5)
        alpha = _definition("Alpha", "class Alpha: ...", symbol_type="class", start_line=10)
        beta = _definition("Beta", "class Beta: ...", symbol_type="class", start_line=20)
        collection = _collection_of(tmp_path, [first, second, alpha, beta])
        context = [_retrieved(first), _retrieved(second)]

        assert referenced_symbols(context, collection) == ["Beta", "Alpha"]
        assert referenced_symbols(context, collection, limit=1) == ["Beta"]


class _ScoringReranker:
    """Scores each candidate by its symbol name, recording every batch it was asked to score."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.batches: list[list[str]] = []

    def rerank(self, query, candidates):
        self.batches.append([c.chunk.symbol_name for c in candidates])
        return [
            RetrievedChunk(chunk=c.chunk, score=self._scores[c.chunk.symbol_name])
            for c in candidates
        ]


class TestRelevantReferences:
    def _index(self, tmp_path):
        requires = _definition(
            "requires", "def requires():\n    WebSocket; RedirectResponse; Scope"
        )
        websocket = _definition("WebSocket", "class WebSocket: ...", start_line=10)
        redirect = _definition("RedirectResponse", "class RedirectResponse: ...", start_line=20)
        scope = _definition("Scope", "class Scope: ...", start_line=30)
        collection = _collection_of(tmp_path, [requires, websocket, redirect, scope])
        return requires, redirect, collection

    def test_offers_only_references_the_reranker_scores_as_relevant_best_first(self, tmp_path):
        requires, _, collection = self._index(tmp_path)
        reranker = _ScoringReranker({"WebSocket": -4.0, "RedirectResponse": 3.0, "Scope": 6.0})
        references = RelevantReferences("how does it redirect?", collection, reranker)

        offered = references([_retrieved(requires)])

        assert [c.chunk.symbol_name for c in offered] == ["Scope", "RedirectResponse"]
        assert [c.score for c in offered] == [6.0, 3.0]

    def test_offers_nothing_when_no_reference_is_relevant(self, tmp_path):
        requires, _, collection = self._index(tmp_path)
        reranker = _ScoringReranker({"WebSocket": -4.0, "RedirectResponse": 0.5, "Scope": -1.0})
        references = RelevantReferences("unrelated question", collection, reranker)

        assert references([_retrieved(requires)]) == []

    def test_scores_each_definition_only_once_across_calls(self, tmp_path):
        requires, redirect, collection = self._index(tmp_path)
        reranker = _ScoringReranker({"WebSocket": -4.0, "RedirectResponse": 3.0, "Scope": 6.0})
        references = RelevantReferences("how does it redirect?", collection, reranker)

        references([_retrieved(requires)])
        after_hop = references([_retrieved(requires), _retrieved(redirect)])

        assert [c.chunk.symbol_name for c in after_hop] == ["Scope"]
        assert [sorted(batch) for batch in reranker.batches] == [
            ["RedirectResponse", "Scope", "WebSocket"],
            [],
        ]
