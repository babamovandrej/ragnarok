import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from src.config import CLONE_TIMEOUT_SECONDS, GIT_COMMAND_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

_GITHUB_HTTPS_URL_PATTERN = re.compile(
    r"^https://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)
_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{7,64}$")


class GitCommandError(RuntimeError):
    """A Git subprocess failed; message is safe to surface, detail is not."""

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        exit_code: int | None = None,
        stderr: str = "",
        repo: str | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.exit_code = exit_code
        self.stderr = stderr
        self.repo = repo


class UnknownRevisionError(GitCommandError):
    """A commit SHA is unreachable in the local clone, e.g. shallow-history limit."""


@dataclass(frozen=True)
class ChangedFile:
    """One entry from a name-status diff between two commits."""

    status: str
    path: str
    previous_path: str | None = None


def _run_git(
    *args: str,
    timeout: int = GIT_COMMAND_TIMEOUT_SECONDS,
    cwd: Path | None = None,
    repo: str | None = None,
) -> str:
    """Run git with prompts disabled, raising GitCommandError instead of a raw subprocess error."""
    operation = args[0] if args else ""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        logger.error("git %s timed out after %ss (repo=%s)", operation, timeout, repo)
        raise GitCommandError(
            f"Git operation '{operation}' timed out", operation=operation, repo=repo
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        logger.error(
            "git %s failed (exit=%s, repo=%s): %s", operation, exc.returncode, repo, stderr
        )
        raise GitCommandError(
            f"Git operation '{operation}' failed",
            operation=operation,
            exit_code=exc.returncode,
            stderr=stderr,
            repo=repo,
        ) from exc
    return result.stdout.strip()


def parse_github_repo(repo_url: str) -> tuple[str, str]:
    """Validate a public GitHub HTTPS URL and split it into (owner, repo)."""
    match = _GITHUB_HTTPS_URL_PATTERN.match(repo_url.strip())
    if match is None:
        raise ValueError(f"Not a supported GitHub repository URL: {repo_url!r}")
    owner, repo = match.group("owner"), match.group("repo")
    if owner in (".", "..") or repo in (".", ".."):
        raise ValueError(f"Not a supported GitHub repository URL: {repo_url!r}")
    return owner, repo


def repo_slug_from_url(repo_url: str) -> str:
    """Return a stable 'owner/repo' identity for cache keys, metadata, and logs."""
    owner, repo = parse_github_repo(repo_url)
    return f"{owner}/{repo}"


def repo_name_from_url(repo_url: str) -> str:
    """Return just the repository name, for display purposes."""
    _, repo = parse_github_repo(repo_url)
    return repo


def _clone_default_branch(repo_url: str, dest: Path, timeout: int) -> None:
    """Shallow-clone a repo's default branch directly."""
    _run_git(
        "clone",
        "--depth",
        "1",
        "--single-branch",
        "--no-tags",
        "--",
        repo_url,
        str(dest),
        timeout=timeout,
        repo=repo_url,
    )


def _clone_revision(repo_url: str, dest: Path, revision: str, timeout: int) -> None:
    """Shallow-fetch a specific branch, tag, or commit SHA into an empty repo at dest."""
    _run_git("init", "--quiet", "--", str(dest), timeout=timeout, repo=repo_url)
    _run_git("remote", "add", "origin", "--", repo_url, cwd=dest, timeout=timeout, repo=repo_url)
    _run_git(
        "fetch", "--depth", "1", "--", "origin", revision, cwd=dest, timeout=timeout, repo=repo_url
    )
    _run_git("checkout", "--quiet", "FETCH_HEAD", cwd=dest, timeout=timeout, repo=repo_url)


def shallow_clone(
    repo_url: str,
    dest: Path,
    timeout: int = CLONE_TIMEOUT_SECONDS,
    revision: str | None = None,
) -> Path:
    """Clone repo_url into dest at depth 1, optionally pinned to a branch/tag/commit."""
    if dest.exists() and any(dest.iterdir()):
        raise ValueError(f"Clone destination is not empty: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Cloning %s into %s%s", repo_url, dest, f" at {revision}" if revision else "")
    started = time.monotonic()
    try:
        if revision is None:
            _clone_default_branch(repo_url, dest, timeout)
        else:
            _clone_revision(repo_url, dest, revision, timeout)
    except GitCommandError:
        if dest.exists():
            try:
                shutil.rmtree(dest)
            except OSError:
                logger.warning("Failed to remove incomplete clone at %s", dest, exc_info=True)
            else:
                logger.info("Removed incomplete clone at %s", dest)
        raise
    logger.info("Cloned %s in %.1fs", repo_url, time.monotonic() - started)
    return dest


def _ensure_git_repository(repo_path: Path) -> None:
    """Raise GitCommandError unless repo_path is a directory containing a Git work tree."""
    if not repo_path.is_dir():
        raise GitCommandError(
            f"Not a directory: {repo_path}", operation="rev-parse", repo=str(repo_path)
        )
    _run_git("rev-parse", "--is-inside-work-tree", cwd=repo_path, repo=str(repo_path))


def _ensure_valid_sha(sha: str) -> None:
    """Reject anything that isn't a plain hexadecimal commit SHA."""
    if not _SHA_PATTERN.match(sha):
        raise ValueError(f"Not a valid Git commit SHA: {sha!r}")


def _ensure_revision_exists(repo_path: Path, sha: str) -> None:
    """Raise UnknownRevisionError if sha is unreachable, e.g. outside a shallow clone's history."""
    try:
        _run_git("cat-file", "-e", f"{sha}^{{commit}}", cwd=repo_path, repo=str(repo_path))
    except GitCommandError as exc:
        logger.warning(
            "Commit %s not found in %s; likely a shallow-clone history limit", sha, repo_path
        )
        raise UnknownRevisionError(
            f"Commit {sha} is not available in this clone (shallow history?)",
            operation="cat-file",
            repo=str(repo_path),
        ) from exc


def current_sha(repo_path: Path) -> str:
    """Return the full 40-character SHA of the repository's current HEAD."""
    _ensure_git_repository(repo_path)
    sha = _run_git("rev-parse", "HEAD", cwd=repo_path, repo=str(repo_path))
    logger.info("Current SHA for %s: %s", repo_path, sha)
    return sha


def _parse_name_status(output: str) -> list[ChangedFile]:
    """Parse null-delimited `git diff --name-status -z` output into ChangedFile entries."""
    tokens = [token for token in output.split("\0") if token]
    changed: list[ChangedFile] = []
    index = 0
    while index < len(tokens):
        status = tokens[index]
        index += 1
        if status[0] in ("R", "C"):
            previous_path, path = tokens[index], tokens[index + 1]
            index += 2
            changed.append(ChangedFile(status=status, path=path, previous_path=previous_path))
        else:
            path = tokens[index]
            index += 1
            changed.append(ChangedFile(status=status, path=path))
    return changed


def changed_files_since(repo_path: Path, since_sha: str) -> list[ChangedFile]:
    """List files added, modified, deleted, renamed, or copied between since_sha and HEAD."""
    _ensure_git_repository(repo_path)
    _ensure_valid_sha(since_sha)
    _ensure_revision_exists(repo_path, since_sha)

    output = _run_git(
        "diff",
        "--name-status",
        "-M",
        "-C",
        "-z",
        since_sha,
        "HEAD",
        cwd=repo_path,
        repo=str(repo_path),
    )
    changed = _parse_name_status(output)
    logger.info("%d changed file(s) since %s in %s", len(changed), since_sha, repo_path)
    return changed
