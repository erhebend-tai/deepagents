"""Small, immutable contracts for opt-in external workers."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class WorkerError(RuntimeError):
    """A safe diagnostic that does not contain credentials or provider bodies."""


class WorkerState(StrEnum):
    """Provider-independent task states; unknown provider states remain running."""

    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def terminal(self) -> bool:
        """Whether the worker can no longer produce changes."""
        return self in {self.COMPLETED, self.FAILED, self.CANCELLED, self.TIMED_OUT}


@dataclass(frozen=True)
class WorkerStatus:
    """An immutable status observation."""

    state: WorkerState


@dataclass(frozen=True)
class WorkerResult:
    """A bounded text result and optional JSON-encoded structured output."""

    state: WorkerState
    output: str = ""
    structured_output: str | None = None


class WorkerAdapter(Protocol):
    """An async adapter owned by one coordinator.

    Implementations must keep handles private, bound operations, and cancel
    resources on close. Local adapters run trusted code, not sandboxed code.
    """

    @property
    def is_local(self) -> bool:
        """Whether this adapter requires an isolated local worktree."""
        ...

    async def submit(self, prompt: str, *, cwd: Path | None) -> str:
        """Start work and return a private provider handle.

        Args:
            prompt: Task instructions.
            cwd: Coordinator-created worktree, or `None` for remote work.
        """
        ...

    async def status(self, handle: str) -> WorkerStatus:
        """Observe work without waiting for completion.

        Args:
            handle: Handle returned by this adapter.
        """
        ...

    async def result(self, handle: str) -> WorkerResult:
        """Retrieve a bounded result.

        Args:
            handle: Handle returned by this adapter.
        """
        ...

    async def cancel(self, handle: str) -> None:
        """Terminate work, including child processes.

        Args:
            handle: Handle returned by this adapter.
        """
        ...

    async def aclose(self) -> None:
        """Release all resources owned by this adapter."""
        ...
