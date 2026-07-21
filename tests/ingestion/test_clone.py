import os
import subprocess
from pathlib import Path

import pytest

from src.ingestion.clone import (
    ChangedFile,
    GitCommandError,
    UnknownRevisionError,
    _run_git,
    changed_files_since,
    current_sha,
    parse_github_repo,
    repo_name_from_url,
    repo_slug_from_url,
    shallow_clone,
)


def _run(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


def _init_git_repo(path: Path) -> None:
    path.mkdir()
    _run("git", "init", "-q", cwd=path)
    _run("git", "config", "user.email", "test@example.com", cwd=path)
    _run("git", "config", "user.name", "Test", cwd=path)


def _init_fixture_repo(path: Path) -> str:
    _init_git_repo(path)

    (path / "a.py").write_text("print('a')\n")
    _run("git", "add", "a.py", cwd=path)
    _run("git", "commit", "-q", "-m", "first", cwd=path)
    first_sha = current_sha(path)

    (path / "a.py").write_text("print('a changed')\n")
    (path / "b.py").write_text("print('b')\n")
    _run("git", "add", "-A", cwd=path)
    _run("git", "commit", "-q", "-m", "second", cwd=path)

    return first_sha


def _write_fake_sleeping_git(bin_dir: Path) -> None:
    bin_dir.mkdir()
    fake_git = bin_dir / "git"
    fake_git.write_text("#!/bin/sh\nsleep 5\n")
    fake_git.chmod(0o755)


class TestRunGit:
    def test_returns_stripped_stdout_on_success(self, tmp_path):
        fixture = tmp_path / "fixture"
        _init_fixture_repo(fixture)

        output = _run_git("rev-parse", "HEAD", cwd=fixture)

        assert len(output) == 40

    def test_raises_git_command_error_on_non_zero_exit(self, tmp_path):
        with pytest.raises(GitCommandError) as exc_info:
            _run_git("not-a-real-git-subcommand")

        assert exc_info.value.exit_code is not None

    def test_captures_stderr_in_error(self, tmp_path):
        with pytest.raises(GitCommandError) as exc_info:
            _run_git("rev-parse", "--is-inside-work-tree", cwd=tmp_path)

        assert exc_info.value.stderr

    def test_raises_on_timeout(self, tmp_path, monkeypatch):
        bin_dir = tmp_path / "fake_bin"
        _write_fake_sleeping_git(bin_dir)
        monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

        with pytest.raises(GitCommandError):
            _run_git("status", timeout=1)

    def test_disables_terminal_prompts(self, monkeypatch):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured.update(kwargs)
            return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        _run_git("status")

        assert captured["env"]["GIT_TERMINAL_PROMPT"] == "0"

    def test_uses_configured_working_directory(self, tmp_path):
        fixture = tmp_path / "fixture"
        _init_fixture_repo(fixture)

        output = _run_git("status", "--short", cwd=fixture)

        assert output == ""


class TestParseGithubRepo:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://github.com/encode/starlette", ("encode", "starlette")),
            ("https://github.com/encode/starlette.git", ("encode", "starlette")),
            ("https://github.com/encode/starlette/", ("encode", "starlette")),
        ],
    )
    def test_accepts_valid_urls(self, url, expected):
        assert parse_github_repo(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "https://github.com/encode/starlette?tab=readme",
            "https://github.com/encode",
            "https://github.com/",
            "git@github.com:encode/starlette.git",
            "file:///home/user/repo",
            "../local-repository",
            "https://gitlab.com/encode/starlette",
            "https://github.com/encode/starlette/tree/main",
            "https://github.com/../..",
            "https://github.com/encode/..",
        ],
    )
    def test_rejects_invalid_urls(self, url):
        with pytest.raises(ValueError):
            parse_github_repo(url)

    def test_repo_slug_from_url_includes_owner(self):
        assert repo_slug_from_url("https://github.com/encode/starlette") == "encode/starlette"

    def test_repo_name_from_url_is_repo_only(self):
        assert repo_name_from_url("https://github.com/encode/starlette") == "starlette"


class TestShallowClone:
    def test_successful_clone_produces_expected_files(self, tmp_path):
        fixture = tmp_path / "fixture"
        _init_fixture_repo(fixture)

        dest = tmp_path / "clone"
        shallow_clone(str(fixture), dest)

        assert (dest / "a.py").is_file()
        assert (dest / "b.py").is_file()
        assert (dest / ".git").is_dir()

    def test_creates_destination_parent_directory(self, tmp_path):
        fixture = tmp_path / "fixture"
        _init_fixture_repo(fixture)

        dest = tmp_path / "nested" / "parent" / "clone"
        shallow_clone(str(fixture), dest)

        assert (dest / "a.py").is_file()

    def test_rejects_non_empty_destination(self, tmp_path):
        fixture = tmp_path / "fixture"
        _init_fixture_repo(fixture)

        dest = tmp_path / "clone"
        dest.mkdir()
        (dest / "existing.txt").write_text("keep me")

        with pytest.raises(ValueError, match="not empty"):
            shallow_clone(str(fixture), dest)
        assert (dest / "existing.txt").exists()

    def test_cleans_up_destination_on_failed_clone(self, tmp_path):
        dest = tmp_path / "clone"

        with pytest.raises(GitCommandError):
            shallow_clone(str(tmp_path / "does-not-exist"), dest)

        assert not dest.exists()

    def test_clones_specific_branch_when_revision_given(self, tmp_path):
        fixture = tmp_path / "fixture"
        _init_git_repo(fixture)
        (fixture / "f.txt").write_text("v1\n")
        _run("git", "add", "-A", cwd=fixture)
        _run("git", "commit", "-q", "-m", "first", cwd=fixture)
        _run("git", "branch", "-q", "feature", cwd=fixture)
        (fixture / "f.txt").write_text("v2\n")
        _run("git", "add", "-A", cwd=fixture)
        _run("git", "commit", "-q", "-m", "second", cwd=fixture)

        dest = tmp_path / "clone"
        shallow_clone(str(fixture), dest, revision="feature")

        assert (dest / "f.txt").read_text() == "v1\n"

    def test_clones_specific_commit_sha_when_revision_given(self, tmp_path):
        fixture = tmp_path / "fixture"
        first_sha = _init_fixture_repo(fixture)

        dest = tmp_path / "clone"
        shallow_clone(str(fixture), dest, revision=first_sha)

        assert (dest / "a.py").read_text() == "print('a')\n"
        assert not (dest / "b.py").exists()

    def test_cleans_up_destination_on_failed_revision_clone(self, tmp_path):
        fixture = tmp_path / "fixture"
        _init_fixture_repo(fixture)

        dest = tmp_path / "clone"
        with pytest.raises(GitCommandError):
            shallow_clone(str(fixture), dest, revision="does-not-exist")

        assert not dest.exists()

    def test_cleans_up_destination_on_timeout(self, tmp_path, monkeypatch):
        bin_dir = tmp_path / "fake_bin"
        _write_fake_sleeping_git(bin_dir)
        monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

        dest = tmp_path / "clone"
        with pytest.raises(GitCommandError):
            shallow_clone("https://github.com/owner/repo", dest, timeout=1)

        assert not dest.exists()


class TestCurrentSha:
    def test_returns_full_hex_sha(self, tmp_path):
        fixture = tmp_path / "fixture"
        _init_fixture_repo(fixture)

        sha = current_sha(fixture)

        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)

    def test_raises_for_non_git_directory(self, tmp_path):
        empty_dir = tmp_path / "not_a_repo"
        empty_dir.mkdir()

        with pytest.raises(GitCommandError):
            current_sha(empty_dir)

    def test_raises_for_missing_path(self, tmp_path):
        with pytest.raises(GitCommandError):
            current_sha(tmp_path / "does-not-exist")

    def test_works_in_detached_head_state(self, tmp_path):
        fixture = tmp_path / "fixture"
        first_sha = _init_fixture_repo(fixture)
        _run("git", "checkout", "-q", first_sha, cwd=fixture)

        assert current_sha(fixture) == first_sha


class TestChangedFilesSince:
    def test_reports_added_and_modified_statuses(self, tmp_path):
        fixture = tmp_path / "fixture"
        first_sha = _init_fixture_repo(fixture)

        changed = changed_files_since(fixture, first_sha)
        by_path = {c.path: c.status for c in changed}

        assert by_path["a.py"] == "M"
        assert by_path["b.py"] == "A"

    def test_reports_deleted_file(self, tmp_path):
        fixture = tmp_path / "fixture"
        _init_git_repo(fixture)
        (fixture / "a.py").write_text("print('a')\n")
        (fixture / "b.py").write_text("print('b')\n")
        _run("git", "add", "-A", cwd=fixture)
        _run("git", "commit", "-q", "-m", "first", cwd=fixture)
        first_sha = current_sha(fixture)

        (fixture / "b.py").unlink()
        _run("git", "add", "-A", cwd=fixture)
        _run("git", "commit", "-q", "-m", "delete b", cwd=fixture)

        changed = changed_files_since(fixture, first_sha)

        assert ChangedFile(status="D", path="b.py") in changed

    def test_reports_renamed_file(self, tmp_path):
        fixture = tmp_path / "fixture"
        _init_git_repo(fixture)
        (fixture / "old_name.py").write_text("def foo():\n    return 1\n" * 5)
        _run("git", "add", "-A", cwd=fixture)
        _run("git", "commit", "-q", "-m", "first", cwd=fixture)
        first_sha = current_sha(fixture)

        _run("git", "mv", "old_name.py", "new_name.py", cwd=fixture)
        _run("git", "commit", "-q", "-m", "rename", cwd=fixture)

        changed = changed_files_since(fixture, first_sha)

        assert len(changed) == 1
        assert changed[0].status.startswith("R")
        assert changed[0].path == "new_name.py"
        assert changed[0].previous_path == "old_name.py"

    def test_reports_copied_file(self, tmp_path):
        fixture = tmp_path / "fixture"
        _init_git_repo(fixture)
        content = "def foo():\n    return 1\n" * 10
        (fixture / "original.py").write_text(content)
        _run("git", "add", "-A", cwd=fixture)
        _run("git", "commit", "-q", "-m", "first", cwd=fixture)
        first_sha = current_sha(fixture)

        (fixture / "original.py").write_text(content + "\n# tweak\n")
        (fixture / "copy.py").write_text(content)
        _run("git", "add", "-A", cwd=fixture)
        _run("git", "commit", "-q", "-m", "copy and tweak", cwd=fixture)

        changed = changed_files_since(fixture, first_sha)
        copy_entries = [c for c in changed if c.status.startswith("C")]

        assert copy_entries
        assert copy_entries[0].path == "copy.py"
        assert copy_entries[0].previous_path == "original.py"

    def test_handles_filenames_with_spaces(self, tmp_path):
        fixture = tmp_path / "fixture"
        first_sha = _init_fixture_repo(fixture)

        (fixture / "new file with spaces.py").write_text("print('x')\n")
        _run("git", "add", "-A", cwd=fixture)
        _run("git", "commit", "-q", "-m", "add spaced file", cwd=fixture)

        changed = changed_files_since(fixture, first_sha)

        assert any(c.path == "new file with spaces.py" for c in changed)

    def test_returns_empty_list_when_nothing_changed(self, tmp_path):
        fixture = tmp_path / "fixture"
        _init_fixture_repo(fixture)
        sha = current_sha(fixture)

        assert changed_files_since(fixture, sha) == []

    def test_rejects_malformed_sha(self, tmp_path):
        fixture = tmp_path / "fixture"
        _init_fixture_repo(fixture)

        with pytest.raises(ValueError):
            changed_files_since(fixture, "not-a-sha; rm -rf /")

    def test_raises_unknown_revision_for_unreachable_sha(self, tmp_path):
        fixture = tmp_path / "fixture"
        _init_fixture_repo(fixture)

        unreachable_sha = "a" * 40
        with pytest.raises(UnknownRevisionError):
            changed_files_since(fixture, unreachable_sha)
