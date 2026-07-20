import pytest


@pytest.fixture
def repo_root():
    from pathlib import Path

    return Path(__file__).parent.parent
