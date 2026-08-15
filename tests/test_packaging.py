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
    """Paths git would include, or None when this is not a git checkout.

    Tracked files *plus* untracked ones that are not ignored. The question this
    guard asks is "would a clone get this file", not "has it been committed
    yet" - a newly written file is fine, a file swallowed by .gitignore is not.
    Asking the stricter question would turn the suite red on every new file,
    and a check that is red for benign reasons is one that stops being read.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT, capture_output=True, text=True, check=True,
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


def test_slurm_can_write_its_logs(tracked):
    """Slurm fails a job outright when it cannot open its output file.

    The sbatch scripts use `--output=logs/...`, relative to the submission
    directory. If a fresh clone has no logs/ directory the job dies before it
    starts, and the explanation would have gone into the very file that could
    not be opened - so the failure arrives with no log at all.
    """
    assert "logs/.gitkeep" in tracked, (
        "logs/ must exist in a clone; keep the .gitkeep negation in .gitignore"
    )

    for script in (ROOT / "scripts" / "slurm").glob("*.sbatch"):
        text = script.read_text()
        if "--output=" in text:
            assert "--output=logs/" in text, (
                f"{script.name} writes elsewhere; update this guard or the script"
            )


def test_batch_scripts_do_not_locate_their_setup_through_dollar_zero():
    """Slurm executes a copy of the script from its spool directory.

    `$0` therefore points into /var/spool, not the repository, and sourcing
    `$(dirname "$0")/common.sh` finds nothing. Nothing that follows is defined,
    so the job dies with a cascade of "command not found" (exit 127) that says
    nothing about the cause.
    """
    for script in (ROOT / "scripts" / "slurm").glob("*.sbatch"):
        text = script.read_text()
        assert 'source "$(dirname "$0")' not in text, (
            f"{script.name} resolves its setup through $0; use SLURM_SUBMIT_DIR"
        )
        assert "SLURM_SUBMIT_DIR" in text, (
            f"{script.name} must locate the repository through SLURM_SUBMIT_DIR"
        )
