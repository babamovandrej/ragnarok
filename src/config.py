import os
from pathlib import Path

from tree_sitter_language_pack import SupportedLanguage

CACHE_ROOT = Path(".ragnarok_cache")

DEVICE_ENV_VAR = "RAGNAROK_DEVICE"
DEFAULT_DEVICE: str | None = os.environ.get(DEVICE_ENV_VAR) or None

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

EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
EMBEDDING_MAX_SEQ_LENGTH = 1024
EMBEDDING_BATCH_SIZE = 2
EMBEDDING_QUERY_PROMPT = (
    "Instruct: Given a question about a codebase, retrieve the code that answers it\nQuery:"
)

RERANK_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
RERANK_MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
RERANK_INSTRUCTION = "Given a question about a codebase, judge whether this code helps answer it"
RERANK_MAX_LENGTH = 2048
RERANK_BATCH_SIZE = 2

DEFAULT_RETRIEVAL_TOP_K = 10
HYBRID_RRF_K = 60

MIN_CONTEXT_RELEVANCE_SCORE = 1.0

GENERATION_MODEL_REPO = "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF"
GENERATION_MODEL_FILENAME = "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
GENERATION_MODEL_REVISION = "f74adce6aa16316c625447af059dbebe4983757c"
GENERATION_CONTEXT_WINDOW = 4096
GENERATION_MAX_TOKENS_DEFAULT = 512
GENERATION_CHAT_TEMPLATE_RESERVE_TOKENS = 64
GENERATION_TEMPERATURE = 0.0

AGENT_MAX_HOPS = 3
AGENT_HOP_CANDIDATE_LIMIT = 8

INDEX_SCHEMA_VERSION = 3

MAX_INDEXABLE_CHUNKS = 1500
MAX_SOURCE_BYTES = 15 * 1024 * 1024
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024
MAX_WHOLE_FILE_LINES = 400
MAX_WHOLE_FILE_CHARS = 20_000
LEADING_COMMENT_MAX_GAP_LINES = 1
CLONE_TIMEOUT_SECONDS = 60
GIT_COMMAND_TIMEOUT_SECONDS = 30

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
        "tests",
        "test",
        "__tests__",
    }
)

DEMO_REPO_URL = "https://github.com/encode/starlette"
