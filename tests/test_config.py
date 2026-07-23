import re

from src import config

_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class TestLanguageExtensionMapping:
    def test_source_extensions_matches_the_language_mapping_keys(self):
        assert config.SOURCE_EXTENSIONS == frozenset(config.LANGUAGE_BY_EXTENSION)

    def test_every_whole_file_language_is_reachable_from_an_extension(self):
        mapped_languages = set(config.LANGUAGE_BY_EXTENSION.values())

        assert config.WHOLE_FILE_LANGUAGES <= mapped_languages

    def test_every_extension_starts_with_a_dot(self):
        assert all(ext.startswith(".") for ext in config.LANGUAGE_BY_EXTENSION)


class TestPinnedModelRevisions:
    def test_every_pinned_revision_looks_like_a_full_git_sha(self):
        revisions = [
            config.EMBEDDING_MODEL_REVISION,
            config.RERANK_MODEL_REVISION,
            config.GENERATION_MODEL_REVISION,
            config.ZEROGPU_MODEL_REVISION,
        ]

        assert all(_GIT_SHA_PATTERN.match(revision) for revision in revisions)


class TestResourceLimitsArePositive:
    def test_retrieval_and_agent_limits_are_positive(self):
        assert config.DEFAULT_RETRIEVAL_TOP_K > 0
        assert config.HYBRID_RRF_K > 0
        assert config.AGENT_MAX_HOPS > 0
        assert config.AGENT_HOP_DECISION_MAX_TOKENS > 0

    def test_ingestion_ceilings_are_positive(self):
        assert config.MAX_INDEXABLE_CHUNKS > 0
        assert config.MAX_SOURCE_BYTES > 0
        assert config.MAX_FILE_SIZE_BYTES > 0
        assert config.MAX_FILE_SIZE_BYTES <= config.MAX_SOURCE_BYTES
        assert config.CLONE_TIMEOUT_SECONDS > 0
        assert config.GIT_COMMAND_TIMEOUT_SECONDS > 0

    def test_max_resident_repos_is_at_least_one(self):
        assert config.MAX_RESIDENT_REPOS >= 1


class TestExcludedDirectoryNames:
    def test_is_immutable(self):
        assert isinstance(config.EXCLUDED_DIRECTORY_NAMES, frozenset)

    def test_includes_common_vcs_and_dependency_directories(self):
        assert {".git", "node_modules", "__pycache__"} <= config.EXCLUDED_DIRECTORY_NAMES
