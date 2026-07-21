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

INDEX_SCHEMA_VERSION = 1

MAX_INDEXABLE_CHUNKS = 1500
MAX_SOURCE_BYTES = 15 * 1024 * 1024
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024
MAX_WHOLE_FILE_LINES = 400
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
    }
)

DEMO_REPO_URL = "https://github.com/encode/starlette"
