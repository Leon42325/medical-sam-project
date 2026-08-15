"""Guards against the repository and the working tree drifting apart.

Motivated by a real failure: `.gitignore` contained the pattern `data/`, meant
for the top-level dataset directory. Git patterns without a leading slash match
at *any* depth, so it also matched `src/samed/data/` and silently excluded that
entire package from version control. Everything passed locally, the push
succeeded, and the omission only surfaced as an ImportError on the cluster.

`git add` never warns about this, so the check has to be a test.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tracked_files() -> set[str] | None:
    """Paths git tracks, or None when this is not a git checkout."""
    try:
        result = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return set(result.stdout.split())


@pytest.fixture(scope="module")
def tracked() -> set[str]:
    files = _tracked_files()
    if files is None:
        pytest.skip("not a git checkout")
    return files


def test_every_source_file_is_committed(tracked):
    on_disk = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "src").rglob("*.py")
        if "egg-info" not in path.parts and "__pycache__" not in path.parts
    }
    missing = sorted(on_disk - tracked)
    assert not missing, (
        "these source files exist locally but are not in the repository, so a "
        f"fresh clone would fail to import them: {missing}"
    )


def test_every_test_and_config_is_committed(tracked):
    on_disk = {
        str(path.relative_to(ROOT))
        for pattern in ("tests/*.py", "configs/*.yaml", "scripts/**/*")
        for path in ROOT.glob(pattern)
        if path.is_file() and "__pycache__" not in path.parts
    }
    missing = sorted(on_disk - tracked)
    assert not missing, f"not in the repository: {missing}"


#: Directory patterns that are *meant* to match at any depth. The distinction
#: is what the name refers to: build and cache artefacts appear throughout the
#: tree and should be ignored everywhere, whereas a pattern naming a project
#: directory ("data", "logs", "results") means the one at the root.
RECURSIVE_BY_DESIGN = frozenset({"__pycache__/"})


def test_gitignore_anchors_its_directory_patterns():
    """An unanchored directory pattern matches at every depth - the original bug."""
    unanchored = [
        line for raw in (ROOT / ".gitignore").read_text().splitlines()
        if (line := raw.strip())
        and not line.startswith(("#", "!", "/", "*", "."))
        and line.endswith("/")
        and line not in RECURSIVE_BY_DESIGN
    ]
    assert not unanchored, (
        "these patterns match a directory of that name at any depth; anchor them "
        f"with a leading slash if only the top-level one is meant: {unanchored}"
    )


def test_the_paper_is_not_committed(tracked):
    """It is copyrighted (Elsevier) and must never reach a public repository."""
    assert not [name for name in tracked if name.lower().endswith(".pdf")]
