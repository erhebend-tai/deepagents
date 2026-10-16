"""Installed Copilot CLI adapter with bounded output and process-group cleanup."""

import asyncio
import contextlib
import math
import os
import shutil
import signal
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from deepagents.workers.types import WorkerError, WorkerResult, WorkerState, WorkerStatus

_MAX_OUTPUT = 65_536
_ENVIRONMENT = frozenset({"PATH", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "SYSTEMROOT", "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"})


@dataclass
class _Process:
    process: asyncio.subprocess.Process
    output: bytearray = field(default_factory=bytearray, repr=False)
    state: WorkerState = WorkerState.RUNNING
    monitor: asyncio.Task[None] | None = None
    reader: asyncio.Task[None] | None = None
    secrets: tuple[str, ...] = field(default=(), repr=False)


async def _terminate(process: asyncio.subprocess.Process) -> None:
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    elif process.returncode is None:
        process.kill()
    await process.wait()


async def _drain(job: _Process) -> None:
    if job.process.stdout is None:
        return
    while chunk := await job.process.stdout.read(8192):
        job.output.extend(chunk)
        del job.output[:-_MAX_OUTPUT]


class CopilotWorker:
    """Run trusted Copilot CLI tasks with read/write permissions, not shell access.

    !!! warning "Experimental"
        A worktree is isolation for changes, NOT a security sandbox. The CLI
        inherits the host's OS access and user credential store through `HOME`.
        Use a container/VM and dedicated credentials for untrusted work.

    Args:
        executable: Installed CLI executable selected by the host, never a model.
        timeout: Finite total task deadline in seconds.
        allowed_tools: Explicit subset of `read` and `write`; no shell permission.
    """

    is_local = True

    def __init__(self, *, executable: str = "copilot", timeout: float = 600, allowed_tools: tuple[str, ...] = ("read", "write")) -> None:
        """Configure an installed CLI without starting any process."""
        if not math.isfinite(timeout) or timeout <= 0:
            msg = "Copilot timeout must be finite and positive."
            raise ValueError(msg)
        if not set(allowed_tools) <= {"read", "write"}:
            msg = "Copilot permissions must be an explicit subset of read and write."
            raise ValueError(msg)
        self._executable = executable
        self._timeout = timeout
        self._allowed_tools = allowed_tools
        self._jobs: dict[str, _Process] = {}
        self._closed = False

    async def _watch(self, job: _Process) -> None:
        state = WorkerState.RUNNING
        try:
            code = await asyncio.wait_for(job.process.wait(), self._timeout)
            state = WorkerState.COMPLETED if code == 0 else WorkerState.FAILED
        except TimeoutError:
            state = WorkerState.TIMED_OUT
        except asyncio.CancelledError:
            state = WorkerState.CANCELLED
            raise
        finally:
            await _terminate(job.process)
            if job.reader is not None:
                await job.reader
            job.state = state

    async def submit(self, prompt: str, *, cwd: Path | None) -> str:
        """Start a CLI process in the supplied isolated worktree.

        Args:
            prompt: A single argument, never interpolated into a shell command.
            cwd: Coordinator-owned detached worktree.
        """
        executable = shutil.which(self._executable)
        if self._closed or cwd is None or executable is None:
            msg = "Copilot worker is closed, missing its workspace, or not installed."
            raise WorkerError(msg)
        arguments = [executable, "-p", prompt]
        for permission in self._allowed_tools:
            arguments.extend(("--allow-tool", permission))
        environment = {key: value for key, value in os.environ.items() if key in _ENVIRONMENT}
        pending = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *arguments,
                cwd=cwd,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
        )
        try:
            process = await asyncio.shield(pending)
        except asyncio.CancelledError:
            process = await pending
            reader = asyncio.create_task(_drain(_Process(process)))
            await _terminate(process)
            await reader
            raise
        except OSError:
            msg = "Copilot could not be started."
            raise WorkerError(msg) from None
        handle = uuid4().hex
        secrets = tuple(environment[key] for key in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN") if environment.get(key))
        job = _Process(process, secrets=secrets)
        self._jobs[handle] = job
        job.reader = asyncio.create_task(_drain(job))
        job.monitor = asyncio.create_task(self._watch(job))
        return handle

    def _job(self, handle: str) -> _Process:
        if handle not in self._jobs:
            msg = "Unknown Copilot task."
            raise WorkerError(msg)
        return self._jobs[handle]

    async def status(self, handle: str) -> WorkerStatus:
        """Observe a CLI task.

        Args:
            handle: Private handle returned by `submit`.
        """
        return WorkerStatus(self._job(handle).state)

    async def result(self, handle: str) -> WorkerResult:
        """Return bounded UTF-8 output without waiting.

        Args:
            handle: Private handle returned by `submit`.
        """
        job = self._job(handle)
        output = bytes(job.output).decode("utf-8", errors="replace")
        for token in job.secrets:
            output = output.replace(token, "[redacted]")
        return WorkerResult(job.state, output[-_MAX_OUTPUT:])

    async def cancel(self, handle: str) -> None:
        """Kill the task's process group and await its output reader.

        Args:
            handle: Private handle returned by `submit`.
        """
        job = self._job(handle)
        if job.monitor is not None and not job.monitor.done():
            job.monitor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await job.monitor
        # Also handles cancellation before the monitor's first scheduling turn.
        await _terminate(job.process)
        if job.reader is not None:
            await job.reader
        if not job.state.terminal:
            job.state = WorkerState.CANCELLED

    async def aclose(self) -> None:
        """Cancel every owned process, including completed tasks' descendants."""
        self._closed = True
        for handle in self._jobs:
            await self.cancel(handle)
