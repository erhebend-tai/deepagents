"""Private, project-local repositories for worker lifecycle tests."""

import contextlib
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest


def git(repository: Path, *arguments: str) -> str:
    return subprocess.run(  # noqa: S603  # Tests construct all arguments and use private repositories.
        [  # noqa: S607  # Existing Git executable.
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "user.name=Worker Test",
            "-c",
            "user.email=worker@example.invalid",
            "-C",
            str(repository),
            *arguments,
        ],
        capture_output=True,
        check=True,
        text=True,
    ).stdout


@pytest.fixture
def workspace():
    parent = Path.cwd() / ".worker-test-artifacts"
    parent.mkdir(exist_ok=True)
    directory = parent / uuid4().hex
    directory.mkdir()
    try:
        yield directory
    finally:
        shutil.rmtree(directory)
        with contextlib.suppress(OSError):
            parent.rmdir()


@pytest.fixture
def repository(workspace):
    directory = workspace / "main"
    directory.mkdir()
    git(directory, "init", "-q")
    (directory / "hello.py").write_text("original\n")
    git(directory, "add", ".")
    git(directory, "commit", "-qm", "initial")
    return directory
