"""Per-session coordination and explicitly approved local patch integration."""

import asyncio
import hashlib
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ParamSpec, Self, TypeVar
from uuid import uuid4

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.config import get_config
from langgraph.types import interrupt

from deepagents.workers.types import WorkerAdapter, WorkerError, WorkerState

_MAX_PATCH = 1_048_576
_MAX_PROMPT = 100_000
_GIT_TIMEOUT = 30
_P = ParamSpec("_P")
_T = TypeVar("_T")


async def _blocking(function: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs) -> _T:
    """Finish bounded filesystem work before allowing cancellation cleanup."""
    operation = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        await operation
        raise


def _git(repository: Path, *arguments: str, index: Path | None = None, patch: bytes | None = None) -> bytes:
    environment = {key: value for key, value in os.environ.items() if key in {"PATH", "HOME", "SYSTEMROOT"}}
    environment.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_TERMINAL_PROMPT": "0"})
    if index is not None:
        environment["GIT_INDEX_FILE"] = str(index)
    try:
        completed = subprocess.run(  # noqa: S603  # Host-owned repository and internally constructed Git arguments, no shell.
            ["git", "-c", f"core.hooksPath={os.devnull}", "-c", "core.fsmonitor=false", "-C", str(repository), *arguments],  # noqa: S607  # Git is the existing host executable.
            input=patch,
            capture_output=True,
            env=environment,
            timeout=_GIT_TIMEOUT,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        msg = "Worker Git operation failed."
        raise WorkerError(msg) from None
    return completed.stdout


@dataclass(frozen=True)
class _Snapshot:
    patch: bytes
    digest: str
    target: str


@dataclass
class _Task:
    worker: str
    handle: str
    worktree: Path | None
    cancelled: bool = False
    cancel_requested: bool = False
    reviewed: _Snapshot | None = None
    integrated: bool = False


class WorkerCoordinator:
    """Own opt-in workers for exactly one live application session.

    !!! warning "Experimental"
        State is in memory, NOT durable across restarts. Keep this instance alive
        while a checkpointed graph is interrupted; use a unique instance per
        session. Close it before exiting to terminate workers and remove trees.

        Local worktrees prevent accidental change overlap, not malicious access.
        Workers share OS permissions and Git objects with the host. Use trusted
        workers or an external container/VM, and keep the main tree quiescent
        during review/integration. Git clean/smudge filters can execute trusted
        repository configuration. Remote output is never executed or applied.

    Integration requires a successful local task, a reviewed snapshot, an
    initially clean main tree at the original `HEAD`, and a checkpointed
    `interrupt()` approval containing `{"approved": True, "task_id": ...,
    "digest": ...}`. Subsequent integrations permit only this coordinator's
    earlier approved changes, with an unchanged real index. Approval digests
    bind patch bytes and the exact expected target tree. Review each patch
    again after another integration; external changes invalidate approval.
    This never merges, commits, pushes, fetches, or stages the real index.

    Args:
        repository: Host-selected Git repository root.
        workers: Host-configured named adapters owned by this coordinator.
        workspace_root: Host-selected parent for private workspaces, outside the
            repository. Defaults to the repository's parent directory.
    """

    def __init__(self, repository: Path, workers: Mapping[str, WorkerAdapter], *, workspace_root: Path | None = None) -> None:
        """Capture a fixed base commit and allocate a private workspace root."""
        self._repository = repository.resolve()
        actual = Path(os.fsdecode(_git(self._repository, "rev-parse", "--show-toplevel")).strip()).resolve()
        if actual != self._repository:
            msg = "The coordinator requires a Git repository root."
            raise WorkerError(msg)
        parent = (workspace_root or self._repository.parent).resolve()
        if parent.is_relative_to(self._repository):
            msg = "Worker workspaces must be outside the main repository."
            raise WorkerError(msg)
        self._base = _git(self._repository, "rev-parse", "HEAD").decode("ascii").strip()
        self._base_tree = _git(self._repository, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
        self._expected_tree = self._base_tree
        self._root = parent / f".deepagents-workers-{uuid4().hex}"
        self._root.mkdir(mode=0o700, parents=True)
        self._workers = dict(workers)
        self._tasks: dict[str, _Task] = {}
        self._worktrees: set[Path] = set()
        self._lock = asyncio.Lock()
        self._closed = False

    async def __aenter__(self) -> Self:
        """Enter the coordinator's resource lifetime."""
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Terminate workers and release worktrees."""
        await self.aclose()

    def _task(self, task_id: str) -> _Task:
        if self._closed or task_id not in self._tasks:
            msg = "Unknown task: not tracked by this coordinator/process, or coordinator closed."
            raise WorkerError(msg)
        return self._tasks[task_id]

    async def submit_worker_task(self, worker: str, prompt: str) -> dict[str, str]:
        """Start a configured worker; return only a coordinator-owned task ID.

        Args:
            worker: Name from the host's configured worker mapping.
            prompt: Task instructions; never a command or workspace selector.
        """
        async with self._lock:
            if self._closed or worker not in self._workers or not prompt.strip() or len(prompt) > _MAX_PROMPT:
                msg = "Unknown worker, closed coordinator, or invalid prompt."
                raise WorkerError(msg)
            task_id = uuid4().hex
            adapter = self._workers[worker]
            worktree = self._root / task_id if adapter.is_local else None
            try:
                if worktree is not None:
                    self._worktrees.add(worktree)
                    await _blocking(_git, self._repository, "worktree", "add", "--detach", str(worktree), self._base)
                handle = await adapter.submit(prompt, cwd=worktree)
            except BaseException:
                # Cancellation is a resource-cleanup path; never suppress it.
                if worktree is not None:
                    await _blocking(self._remove_worktree, worktree)
                    self._worktrees.discard(worktree)
                raise
            self._tasks[task_id] = _Task(worker, handle, worktree)
            return {"task_id": task_id, "worker": worker}

    async def check_worker_status(self, task_id: str) -> dict[str, str]:
        """Observe a task without blocking for completion.

        Args:
            task_id: Opaque ID returned by `submit_worker_task`.
        """
        task = self._task(task_id)
        state = WorkerState.CANCELLED if task.cancelled else (await self._workers[task.worker].status(task.handle)).state
        return {"task_id": task_id, "worker": task.worker, "state": state.value}

    async def get_worker_result(self, task_id: str) -> dict[str, str | None]:
        """Read bounded output, treating worker text as untrusted data.

        Args:
            task_id: Opaque ID returned by `submit_worker_task`.
        """
        task = self._task(task_id)
        if task.cancelled:
            return {"task_id": task_id, "state": WorkerState.CANCELLED.value, "output": "", "structured_output": None}
        result = await self._workers[task.worker].result(task.handle)
        return {"task_id": task_id, "state": result.state.value, "output": result.output, "structured_output": result.structured_output}

    async def cancel_worker_task(self, task_id: str) -> dict[str, str]:
        """Request termination and report observed state, not assumed success.

        Args:
            task_id: Opaque ID returned by `submit_worker_task`.
        """
        async with self._lock:
            task = self._task(task_id)
            if not task.cancel_requested:
                await self._workers[task.worker].cancel(task.handle)
                task.cancel_requested = True
            state = (await self._workers[task.worker].status(task.handle)).state
            task.cancelled = state is WorkerState.CANCELLED
            return {"task_id": task_id, "state": state.value}

    async def list_worker_tasks(self) -> list[dict[str, str]]:
        """List observations of this coordinator's tasks only."""
        return [await self.check_worker_status(task_id) for task_id in self._tasks]

    async def _completed_local(self, task_id: str) -> _Task:
        task = self._task(task_id)
        state = (await self.check_worker_status(task_id))["state"]
        if task.worktree is None or state != WorkerState.COMPLETED.value or task.integrated or task.cancel_requested:
            msg = "Only completed, unintegrated local tasks can be reviewed or integrated."
            raise WorkerError(msg)
        return task

    @contextmanager
    def _index(self, repository: Path, tree: str) -> Iterator[Path]:
        index = self._root / f"index-{uuid4().hex}.index"
        try:
            _git(repository, "read-tree", tree, index=index)
            yield index
        finally:
            index.unlink(missing_ok=True)
            index.with_suffix(".index.lock").unlink(missing_ok=True)

    def _snapshot(self, task: _Task) -> _Snapshot:
        if task.worktree is None:
            msg = "Remote results cannot be imported as local patches."
            raise WorkerError(msg)
        with self._index(task.worktree, self._base) as index:
            _git(task.worktree, "add", "-A", "--", ".", index=index)
            patch = _git(
                task.worktree, "diff", "--cached", "--binary", "--full-index", "--no-ext-diff", "--no-textconv", self._base, "--", index=index
            )
        if len(patch) > _MAX_PATCH or re.search(rb"^(?:index .* |(?:new file|deleted file|old|new) mode )160000$", patch, re.MULTILINE):
            msg = "Worker patch is too large or contains unsupported submodule changes."
            raise WorkerError(msg)
        try:
            patch.decode("utf-8")
        except UnicodeDecodeError:
            msg = "Worker patch must be UTF-8 text or a Git binary patch for exact review."
            raise WorkerError(msg) from None
        digest = hashlib.sha256(self._base.encode() + b"\0" + self._expected_tree.encode() + b"\0" + patch).hexdigest()
        return _Snapshot(patch, digest, self._expected_tree)

    async def review_worker_changes(self, task_id: str) -> dict[str, str]:
        """Capture the exact local diff, including additions and worker commits.

        Args:
            task_id: Opaque ID of a completed local task.
        """
        async with self._lock:
            task = await self._completed_local(task_id)
            await _blocking(self._verify_main)
            task.reviewed = await _blocking(self._snapshot, task)
            return {"task_id": task_id, "digest": task.reviewed.digest, "diff": task.reviewed.patch.decode("utf-8")}

    def _verify_main(self) -> None:
        head = _git(self._repository, "rev-parse", "HEAD").decode("ascii").strip()
        staged = _git(self._repository, "diff", "--cached", "--name-only", self._base, "--")
        with self._index(self._repository, self._base) as index:
            _git(self._repository, "add", "-A", "--", ".", index=index)
            tree = _git(self._repository, "write-tree", index=index).decode("ascii").strip()
        if head != self._base or staged or tree != self._expected_tree:
            msg = "Integration requires the original HEAD, a clean main tree or only prior approved changes, and an unchanged index."
            raise WorkerError(msg)

    def _apply(self, task: _Task, snapshot: _Snapshot) -> None:
        self._verify_main()
        if snapshot.target != self._expected_tree:
            msg = "The target tree changed since review; review again."
            raise WorkerError(msg)
        if snapshot.patch:
            with self._index(self._repository, self._expected_tree) as index:
                _git(self._repository, "apply", "--cached", "--binary", patch=snapshot.patch, index=index)
                expected = _git(self._repository, "write-tree", index=index).decode("ascii").strip()
            _git(self._repository, "apply", "--check", "--binary", patch=snapshot.patch)
            self._verify_main()
            _git(self._repository, "apply", "--binary", patch=snapshot.patch)
            self._expected_tree = expected
            self._verify_main()
        task.integrated = True

    async def integrate_worker_changes(self, task_id: str) -> dict[str, str]:
        """Interrupt for host approval, then apply only the reviewed local patch.

        Args:
            task_id: Opaque ID of a completed and reviewed local task.
        """
        async with self._lock:
            task = await self._completed_local(task_id)
            snapshot = task.reviewed
            if snapshot is None:
                msg = "Review worker changes before requesting integration."
                raise WorkerError(msg)
            try:
                configured = get_config().get("configurable", {}).get("__pregel_checkpointer")
            except RuntimeError:
                configured = None
            if not configured:
                msg = "Integration requires a checkpointed LangGraph host approval."
                raise WorkerError(msg)
            current = await _blocking(self._snapshot, task)
            if current.digest != snapshot.digest:
                msg = "Worker or target changes differ from the reviewed snapshot; review again."
                raise WorkerError(msg)
            await _blocking(self._verify_main)
            approval = interrupt(
                {
                    "action": "integrate_worker_changes",
                    "task_id": task_id,
                    "digest": snapshot.digest,
                    "target_tree": snapshot.target,
                    "diff": snapshot.patch.decode("utf-8"),
                }
            )
            if (
                not isinstance(approval, dict)
                or approval.get("approved") is not True
                or approval.get("task_id") != task_id
                or approval.get("digest") != snapshot.digest
            ):
                return {"task_id": task_id, "state": "rejected"}
            # Recheck after approval, and apply the immutable bytes rather than a
            # worker-owned path. Concurrent external main-tree writers are unsupported.
            current = await _blocking(self._snapshot, task)
            if current.digest != snapshot.digest:
                msg = "Worker changed after approval; review again."
                raise WorkerError(msg)
            await _blocking(self._apply, task, snapshot)
            return {"task_id": task_id, "state": "integrated", "digest": snapshot.digest}

    def tools(self) -> list[BaseTool]:
        """Build tools to pass explicitly to `create_deep_agent(tools=...)`."""
        methods = (
            self.submit_worker_task,
            self.check_worker_status,
            self.get_worker_result,
            self.cancel_worker_task,
            self.list_worker_tasks,
            self.review_worker_changes,
            self.integrate_worker_changes,
        )
        return [StructuredTool.from_function(coroutine=method) for method in methods]

    def _remove_worktree(self, worktree: Path) -> None:
        registered = _git(self._repository, "worktree", "list", "--porcelain", "-z").split(b"\0")
        if b"worktree " + os.fsencode(worktree) in registered:
            _git(self._repository, "worktree", "remove", "--force", str(worktree))
        elif worktree.exists():
            shutil.rmtree(worktree)

    async def aclose(self) -> None:
        """Stop all adapters before removing worktrees; retry on cleanup failure."""
        async with self._lock:
            self._closed = True
            failed = False
            for adapter in self._workers.values():
                try:
                    await adapter.aclose()
                except WorkerError:
                    failed = True
            if failed:
                msg = "Worker cleanup failed; worktrees retained. Retry aclose()."
                raise WorkerError(msg)
            for worktree in tuple(self._worktrees):
                await _blocking(self._remove_worktree, worktree)
                self._worktrees.discard(worktree)
            if self._root.exists():
                shutil.rmtree(self._root, ignore_errors=False)
