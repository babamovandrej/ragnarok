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

EMBEDDING_MODEL_ID = "jinaai/jina-embeddings-v2-base-code"
EMBEDDING_MODEL_REVISION = "516f4baf13dec4ddddda8631e019b5737c8bc250"
EMBEDDING_MAX_SEQ_LENGTH = 2048
EMBEDDING_BATCH_SIZE = 8

RERANK_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_MODEL_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"

DEFAULT_RETRIEVAL_TOP_K = 10
HYBRID_RRF_K = 60

GENERATION_MODEL_REPO = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
GENERATION_MODEL_FILENAME = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
GENERATION_MODEL_REVISION = "f86cb2c1fa58255f8052cc32aeede1b7482d4361"
GENERATION_CONTEXT_WINDOW = 4096
GENERATION_MAX_TOKENS_DEFAULT = 512

AGENT_MAX_HOPS = 3
AGENT_HOP_DECISION_MAX_TOKENS = 32

INDEX_SCHEMA_VERSION = 1

MAX_INDEXABLE_CHUNKS = 1500
MAX_SOURCE_BYTES = 15 * 1024 * 1024
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024
MAX_WHOLE_FILE_LINES = 400
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
