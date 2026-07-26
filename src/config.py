from pathlib import Path

from tree_sitter_language_pack import SupportedLanguage

CACHE_ROOT = Path(".ragnarok_cache")

LANGUAGE_BY_EXTENSION: dict[str, SupportedLanguage] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
}

SOURCE_EXTENSIONS: frozenset[str] = frozenset(LANGUAGE_BY_EXTENSION)

WHOLE_FILE_LANGUAGES: frozenset[SupportedLanguage] = frozenset({"yaml", "json", "toml"})

# Tops MTEB code/multilingual retrieval, Apache-2.0, natively sentence-transformers-loadable.
# Revision verified via the HF API, not fabricated. Not a drop-in swap: changing this changes
# embedding_dimension, so every existing index becomes incompatible and must be rebuilt from
# scratch (`python -m scripts.build_index <repo-url> --rebuild`) — see README.md.
EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-4B"
EMBEDDING_MODEL_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
EMBEDDING_MAX_SEQ_LENGTH = 2048
EMBEDDING_BATCH_SIZE = 8

# Despite being a causal-LM-backed yes/no-token reranker internally, this model's HF model
# card documents direct `sentence_transformers.CrossEncoder(...).predict(...)` usage
# identical to this codebase's existing call pattern in rerank.py. Revision verified via the
# HF API, not fabricated.
RERANK_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
RERANK_MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"

DEFAULT_RETRIEVAL_TOP_K = 10
HYBRID_RRF_K = 60

# Below this reranked relevance score, compose_answer_node treats the question as not covered
# by the indexed repo and answers without calling the LLM at all. RERANK_MODEL_ID's
# cross-encoder scores are unbounded raw logits, not 0-1 probabilities; 0.0 is a conservative
# placeholder, not empirically calibrated — recalibrate against the adversarial "not in this
# repo" items in data/eval/qa_starlette.jsonl once the eval harness is actually run (task 15).
MIN_CONTEXT_RELEVANCE_SCORE = 0.0

GENERATION_MODEL_REPO = "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF"
GENERATION_MODEL_FILENAME = "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
GENERATION_MODEL_REVISION = "f74adce6aa16316c625447af059dbebe4983757c"
GENERATION_CONTEXT_WINDOW = 4096
GENERATION_MAX_TOKENS_DEFAULT = 512

ZEROGPU_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
ZEROGPU_MODEL_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"

AGENT_MAX_HOPS = 3
# 64, not 32: the hop-decision reply now carries a SYMBOL: line plus a REASON: line (see
# task 11 / graph.py's DECIDE_MORE_HOPS_PROMPT) so HopRecord.reason can reflect the model's
# actual justification instead of a hardcoded template.
AGENT_HOP_DECISION_MAX_TOKENS = 64

# Bump whenever a change requires rejecting/rebuilding indexes built under the old schema.
# v2: added a required per-chunk repo_sha field to Chroma metadata (see task 09) — indexes
# built under v1 don't have it, so chunk_from_chroma_record would KeyError on retrieval
# against them if this weren't bumped, since is_compatible_with()/has_compatible_embedding_
# setup() are what force a rebuild instead of silently reusing an incompatible index.
INDEX_SCHEMA_VERSION = 2

MAX_INDEXABLE_CHUNKS = 1500
MAX_SOURCE_BYTES = 15 * 1024 * 1024
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024
MAX_WHOLE_FILE_LINES = 400
# A second, independent cap alongside MAX_WHOLE_FILE_LINES: a single line-count-bounded
# "line" can still be unbounded in size (a minified JS file, a generated single-line JSON
# blob), so _whole_file_chunk also re-splits by raw character count.
MAX_WHOLE_FILE_CHARS = 20_000
LEADING_COMMENT_MAX_GAP_LINES = 1
CLONE_TIMEOUT_SECONDS = 60
GIT_COMMAND_TIMEOUT_SECONDS = 30

MAX_RESIDENT_REPOS = 2

EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".next",
        ".nuxt",
        "dist",
        "build",
        "coverage",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".idea",
        ".vscode",
        "target",
        "vendor",
        "bin",
        "obj",
    }
)

DEMO_REPO_URL = "https://github.com/encode/starlette"
