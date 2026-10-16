"""Real Git isolation, checkpointed approval, and stale snapshot rejection."""

import asyncio
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from deepagents.workers import WorkerCoordinator, WorkerError, WorkerResult, WorkerState, WorkerStatus
from tests.unit_tests.workers.conftest import git

if TYPE_CHECKING:
    from pathlib import Path


class FakeWorker:
    def __init__(self, *, local=True, state=WorkerState.COMPLETED, fail=False) -> None:
        self.is_local = local
        self.state = state
        self.fail = fail
        self.workspaces: dict[str, Path | None] = {}
        self.cancelled = []
        self.closed = False

    async def submit(self, prompt, *, cwd):
        if self.fail:
            msg = "Safe startup failure."
            raise WorkerError(msg)
        handle = uuid4().hex
        self.workspaces[handle] = cwd
        return handle

    async def status(self, handle):
        return WorkerStatus(WorkerState.CANCELLED if handle in self.cancelled else self.state)

    async def result(self, handle):
        return WorkerResult(self.state, "bounded output", '{"tests":"passed"}')

    async def cancel(self, handle):
        self.cancelled.append(handle)

    async def aclose(self):
        self.closed = True


@dataclass
class GraphState:
    result: dict[str, str]


def integration_graph(coordinator, task_id, *, checkpointed=True):
    tool = next(tool for tool in coordinator.tools() if tool.name == "integrate_worker_changes")

    async def integrate(state: GraphState) -> dict[str, dict[str, str]]:
        return {"result": state.result | await tool.ainvoke({"task_id": task_id})}

    builder = StateGraph(GraphState)
    builder.add_node("integrate", integrate)
    builder.add_edge(START, "integrate")
    builder.add_edge("integrate", END)
    return builder.compile(checkpointer=InMemorySaver() if checkpointed else None)


async def test_tool_surface_task_visibility_and_detached_isolation(repository):
    worker = FakeWorker()
    baseline = git(repository, "rev-parse", "HEAD")
    async with WorkerCoordinator(repository, {"copilot": worker}) as coordinator:
        tools = {tool.name: tool for tool in coordinator.tools()}
        assert set(tools) == {
            "submit_worker_task",
            "check_worker_status",
            "get_worker_result",
            "cancel_worker_task",
            "list_worker_tasks",
            "review_worker_changes",
            "integrate_worker_changes",
        }
        first = await tools["submit_worker_task"].ainvoke({"worker": "copilot", "prompt": "first"})
        second = await coordinator.submit_worker_task("copilot", "second")
        assert set(first) == {"task_id", "worker"}
        assert first["task_id"] != second["task_id"]
        trees = [tree for tree in worker.workspaces.values() if tree is not None]
        assert len(trees) == 2
        assert all(tree != repository and tree.is_dir() for tree in trees)
        assert all(git(tree, "rev-parse", "HEAD") == baseline for tree in trees)
        assert all(git(tree, "branch", "--show-current") == "" for tree in trees)
        (trees[0] / "hello.py").write_text("worker one\n")
        assert (trees[1] / "hello.py").read_text() == "original\n"
        assert (repository / "hello.py").read_text() == "original\n"
        tasks = await tools["list_worker_tasks"].ainvoke({})
        assert len(tasks) == 2
        result = await coordinator.get_worker_result(first["task_id"])
        assert result["output"] == "bounded output"
        assert result["structured_output"] == '{"tests":"passed"}'
        root = trees[0].parent
    assert worker.closed
    assert not root.exists()
    assert git(repository, "worktree", "list", "--porcelain").count("worktree ") == 1
    assert git(repository, "rev-parse", "HEAD") == baseline


@pytest.mark.parametrize("approved", [True, False])
async def test_host_interrupt_approval_or_rejection(repository, approved):
    worker = FakeWorker()
    async with WorkerCoordinator(repository, {"copilot": worker}) as coordinator:
        task_id = (await coordinator.submit_worker_task("copilot", "change"))["task_id"]
        tree = next(iter(worker.workspaces.values()))
        assert tree is not None
        (tree / "hello.py").write_text("updated\n")
        (tree / "added.py").write_text("new file\n")
        # A worker commit remains reviewable without importing its commit object
        # as a merge or moving the main branch.
        git(tree, "add", ".")
        git(tree, "commit", "-qm", "worker commit")
        review = await coordinator.review_worker_changes(task_id)
        assert "updated" in review["diff"]
        assert "added.py" in review["diff"]
        graph = integration_graph(coordinator, task_id)
        config = {"configurable": {"thread_id": uuid4().hex}}
        paused = await graph.ainvoke({"result": {}}, config)
        request = paused["__interrupt__"][0].value
        assert request["digest"] == review["digest"]
        assert request["diff"] == review["diff"]
        assert (repository / "hello.py").read_text() == "original\n"
        before = git(repository, "rev-parse", "HEAD")
        resumed = await graph.ainvoke(
            Command(
                resume={
                    "approved": approved,
                    "task_id": task_id,
                    "digest": review["digest"],
                }
            ),
            config,
        )
        assert resumed["result"]["state"] == ("integrated" if approved else "rejected")
        assert (repository / "hello.py").read_text() == ("updated\n" if approved else "original\n")
        assert (repository / "added.py").exists() is approved
        assert git(repository, "rev-parse", "HEAD") == before


async def test_integration_requires_review_and_real_checkpointer(repository):
    worker = FakeWorker()
    async with WorkerCoordinator(repository, {"copilot": worker}) as coordinator:
        task_id = (await coordinator.submit_worker_task("copilot", "change"))["task_id"]
        with pytest.raises(WorkerError, match="Review"):
            await coordinator.integrate_worker_changes(task_id)
        await coordinator.review_worker_changes(task_id)
        with pytest.raises(WorkerError, match="checkpointed"):
            await coordinator.integrate_worker_changes(task_id)
        graph = integration_graph(coordinator, task_id, checkpointed=False)
        with pytest.raises(WorkerError, match="checkpointed"):
            await graph.ainvoke({"result": {}})
        assert not git(repository, "status", "--porcelain")


@pytest.mark.parametrize("changed", ["worker", "mode", "main", "head"])
async def test_changed_snapshot_or_main_rejects_resume(repository, changed):
    worker = FakeWorker()
    async with WorkerCoordinator(repository, {"copilot": worker}) as coordinator:
        task_id = (await coordinator.submit_worker_task("copilot", "change"))["task_id"]
        tree = next(iter(worker.workspaces.values()))
        assert tree is not None
        (tree / "hello.py").write_text("reviewed\n")
        review = await coordinator.review_worker_changes(task_id)
        graph = integration_graph(coordinator, task_id)
        config = {"configurable": {"thread_id": uuid4().hex}}
        await graph.ainvoke({"result": {}}, config)
        if changed == "worker":
            (tree / "hello.py").write_text("not approved\n")
        elif changed == "mode":
            (tree / "hello.py").chmod(0o755)
        elif changed == "main":
            (repository / "untracked.py").write_text("user work\n")
        else:
            git(repository, "commit", "--allow-empty", "-qm", "new host head")
        with pytest.raises(WorkerError, match=r"reviewed snapshot|clean main tree"):
            await graph.ainvoke(Command(resume={"approved": True, "task_id": task_id, "digest": review["digest"]}), config)
        assert (repository / "hello.py").read_text() == "original\n"


@pytest.mark.parametrize("response", [True, {"approved": True}, {"approved": True, "task_id": "other", "digest": "wrong"}])
async def test_approval_is_bound_to_exact_task_and_digest(repository, response):
    worker = FakeWorker()
    async with WorkerCoordinator(repository, {"copilot": worker}) as coordinator:
        task_id = (await coordinator.submit_worker_task("copilot", "change"))["task_id"]
        tree = next(iter(worker.workspaces.values()))
        assert tree is not None
        (tree / "hello.py").write_text("updated\n")
        await coordinator.review_worker_changes(task_id)
        graph = integration_graph(coordinator, task_id)
        config = {"configurable": {"thread_id": uuid4().hex}}
        await graph.ainvoke({"result": {}}, config)
        result = await graph.ainvoke(Command(resume=response), config)
        assert result["result"]["state"] == "rejected"
        assert (repository / "hello.py").read_text() == "original\n"


@pytest.mark.parametrize(
    ("local", "state"),
    [
        (False, WorkerState.COMPLETED),
        (True, WorkerState.RUNNING),
        (True, WorkerState.FAILED),
        (True, WorkerState.BLOCKED),
        (True, WorkerState.TIMED_OUT),
    ],
)
async def test_only_successful_local_results_are_integrable(repository, local, state):
    worker = FakeWorker(local=local, state=state)
    async with WorkerCoordinator(repository, {"worker": worker}) as coordinator:
        task_id = (await coordinator.submit_worker_task("worker", "change"))["task_id"]
        with pytest.raises(WorkerError, match="completed"):
            await coordinator.review_worker_changes(task_id)
        with pytest.raises(WorkerError, match="completed"):
            await coordinator.integrate_worker_changes(task_id)


async def test_cancel_and_task_ids_cannot_select_other_paths(repository):
    worker = FakeWorker()
    async with WorkerCoordinator(repository, {"worker": worker}) as coordinator:
        task_id = (await coordinator.submit_worker_task("worker", "change"))["task_id"]
        await coordinator.cancel_worker_task(task_id)
        await coordinator.cancel_worker_task(task_id)
        assert len(worker.cancelled) == 1
        assert (await coordinator.check_worker_status(task_id))["state"] == "cancelled"
        assert (await coordinator.get_worker_result(task_id))["output"] == ""
        with pytest.raises(WorkerError, match="Unknown task"):
            await coordinator.review_worker_changes("../main")
        with pytest.raises(WorkerError, match="completed"):
            await coordinator.review_worker_changes(task_id)


async def test_submission_failure_removes_worktree(repository):
    async with WorkerCoordinator(repository, {"worker": FakeWorker(fail=True)}) as coordinator:
        with pytest.raises(WorkerError, match="startup"):
            await coordinator.submit_worker_task("worker", "change")
        assert await coordinator.list_worker_tasks() == []
        assert git(repository, "worktree", "list", "--porcelain").count("worktree ") == 1


async def test_instances_have_no_global_registry(repository):
    worker = FakeWorker()
    async with WorkerCoordinator(repository, {"worker": worker}) as first, WorkerCoordinator(repository, {"worker": FakeWorker()}) as second:
        task_id = (await first.submit_worker_task("worker", "change"))["task_id"]
        assert await second.list_worker_tasks() == []
        with pytest.raises(WorkerError, match="Unknown task"):
            await second.get_worker_result(task_id)


def test_workspaces_cannot_pollute_main_tree(repository):
    with pytest.raises(WorkerError, match="outside"):
        WorkerCoordinator(repository, {}, workspace_root=repository / "workers")


async def test_cancelled_submission_removes_worktree(repository):
    started = asyncio.Event()

    class WaitingWorker(FakeWorker):
        async def submit(self, prompt, *, cwd):
            started.set()
            await asyncio.Event().wait()
            return "unreachable"

    async with WorkerCoordinator(repository, {"worker": WaitingWorker()}) as coordinator:
        submission = asyncio.create_task(coordinator.submit_worker_task("worker", "change"))
        await asyncio.wait_for(started.wait(), 3)
        submission.cancel()
        with pytest.raises(asyncio.CancelledError):
            await submission
        assert await coordinator.list_worker_tasks() == []
        assert git(repository, "worktree", "list", "--porcelain").count("worktree ") == 1


async def test_close_failure_preserves_worktrees_until_retry(repository):
    class RetryWorker(FakeWorker):
        async def aclose(self):
            if not self.closed:
                self.closed = True
                msg = "Safe cleanup failure."
                raise WorkerError(msg)

    worker = RetryWorker()
    coordinator = WorkerCoordinator(repository, {"worker": worker})
    await coordinator.submit_worker_task("worker", "change")
    tree = next(iter(worker.workspaces.values()))
    assert tree is not None
    with pytest.raises(WorkerError, match="worktrees retained"):
        await coordinator.aclose()
    assert tree.exists()
    await coordinator.aclose()
    assert not tree.exists()
    await coordinator.aclose()


async def test_review_does_not_stage_worker_files(repository):
    worker = FakeWorker()
    async with WorkerCoordinator(repository, {"worker": worker}) as coordinator:
        task_id = (await coordinator.submit_worker_task("worker", "change"))["task_id"]
        tree = next(iter(worker.workspaces.values()))
        assert tree is not None
        (tree / "hello.py").write_text("updated\n")
        (tree / "added.py").write_text("added\n")
        before = git(tree, "status", "--porcelain")
        review = await coordinator.review_worker_changes(task_id)
        assert "added.py" in review["diff"]
        assert git(tree, "status", "--porcelain") == before
        assert git(tree, "diff", "--cached") == ""


async def test_rereview_cannot_reuse_older_approval(repository):
    worker = FakeWorker()
    async with WorkerCoordinator(repository, {"worker": worker}) as coordinator:
        task_id = (await coordinator.submit_worker_task("worker", "change"))["task_id"]
        tree = next(iter(worker.workspaces.values()))
        assert tree is not None
        (tree / "hello.py").write_text("first review\n")
        original = await coordinator.review_worker_changes(task_id)
        graph = integration_graph(coordinator, task_id)
        config = {"configurable": {"thread_id": uuid4().hex}}
        await graph.ainvoke({"result": {}}, config)
        (tree / "hello.py").write_text("second review\n")
        second = await coordinator.review_worker_changes(task_id)
        assert original["digest"] != second["digest"]
        resumed = await graph.ainvoke(
            Command(
                resume={
                    "approved": True,
                    "task_id": task_id,
                    "digest": original["digest"],
                }
            ),
            config,
        )
        assert resumed["result"]["state"] == "rejected"
        assert (repository / "hello.py").read_text() == "original\n"


async def test_close_removes_registration_for_missing_worktree(repository):
    worker = FakeWorker()
    coordinator = WorkerCoordinator(repository, {"worker": worker})
    await coordinator.submit_worker_task("worker", "change")
    tree = next(iter(worker.workspaces.values()))
    assert tree is not None
    shutil.rmtree(tree)
    await coordinator.aclose()
    assert git(repository, "worktree", "list", "--porcelain").count("worktree ") == 1


@pytest.mark.parametrize("unsupported", ["oversized", "non-utf8"])
async def test_unreviewable_patches_are_rejected_not_truncated(repository, monkeypatch, unsupported):
    worker = FakeWorker()
    async with WorkerCoordinator(repository, {"worker": worker}) as coordinator:
        task_id = (await coordinator.submit_worker_task("worker", "change"))["task_id"]
        tree = next(iter(worker.workspaces.values()))
        assert tree is not None
        if unsupported == "oversized":
            monkeypatch.setattr("deepagents.workers.coordinator._MAX_PATCH", 10)
            (tree / "hello.py").write_text("updated\n")
        else:
            (tree / "hello.py").write_bytes(b"non-utf8: \xff\n")
        with pytest.raises(WorkerError, match=r"too large|exact review"):
            await coordinator.review_worker_changes(task_id)
        with pytest.raises(WorkerError, match="Review"):
            await coordinator.integrate_worker_changes(task_id)
        assert (repository / "hello.py").read_text() == "original\n"


async def test_unconfirmed_remote_cancellation_is_not_reported_stopped(repository):
    class PendingRemote(FakeWorker):
        async def cancel(self, handle):
            pass

    worker = PendingRemote(local=False, state=WorkerState.RUNNING)
    async with WorkerCoordinator(repository, {"worker": worker}) as coordinator:
        task_id = (await coordinator.submit_worker_task("worker", "task"))["task_id"]
        cancellation = await coordinator.cancel_worker_task(task_id)
        assert cancellation["state"] == "running"
        assert (await coordinator.check_worker_status(task_id))["state"] == "running"


async def approve_patch(coordinator, task_id):
    review = await coordinator.review_worker_changes(task_id)
    graph = integration_graph(coordinator, task_id)
    config = {"configurable": {"thread_id": uuid4().hex}}
    paused = await graph.ainvoke({"result": {}}, config)
    request = paused["__interrupt__"][0].value
    assert request["digest"] == review["digest"]
    return await graph.ainvoke(
        Command(
            resume={
                "approved": True,
                "task_id": task_id,
                "digest": request["digest"],
            }
        ),
        config,
    )


async def test_multiple_approved_patches_accumulate_without_commits_or_staging(repository):
    worker = FakeWorker()
    original = git(repository, "rev-parse", "HEAD")
    async with WorkerCoordinator(repository, {"worker": worker}) as coordinator:
        first = (await coordinator.submit_worker_task("worker", "first"))["task_id"]
        second = (await coordinator.submit_worker_task("worker", "second"))["task_id"]
        trees = [tree for tree in worker.workspaces.values() if tree is not None]
        (trees[0] / "hello.py").write_text("first approved change\n")
        (trees[1] / "second.py").write_text("second approved change\n")
        first_result = await approve_patch(coordinator, first)
        second_result = await approve_patch(coordinator, second)
        assert first_result["result"]["state"] == "integrated"
        assert second_result["result"]["state"] == "integrated"
        assert (repository / "hello.py").read_text() == "first approved change\n"
        assert (repository / "second.py").read_text() == "second approved change\n"
        assert git(repository, "rev-parse", "HEAD") == original
        assert git(repository, "diff", "--cached") == ""


async def test_prior_integration_invalidates_pending_target_approval(repository):
    worker = FakeWorker()
    async with WorkerCoordinator(repository, {"worker": worker}) as coordinator:
        first = (await coordinator.submit_worker_task("worker", "first"))["task_id"]
        second = (await coordinator.submit_worker_task("worker", "second"))["task_id"]
        trees = [tree for tree in worker.workspaces.values() if tree is not None]
        (trees[0] / "first.py").write_text("first\n")
        (trees[1] / "second.py").write_text("second\n")
        review = await coordinator.review_worker_changes(second)
        graph = integration_graph(coordinator, second)
        config = {"configurable": {"thread_id": uuid4().hex}}
        await graph.ainvoke({"result": {}}, config)
        await approve_patch(coordinator, first)
        with pytest.raises(WorkerError, match="reviewed snapshot"):
            await graph.ainvoke(Command(resume={"approved": True, "task_id": second, "digest": review["digest"]}), config)
        refreshed = await coordinator.review_worker_changes(second)
        assert refreshed["diff"] == review["diff"]
        assert refreshed["digest"] != review["digest"]
        assert not (repository / "second.py").exists()
        assert (await approve_patch(coordinator, second))["result"]["state"] == "integrated"


@pytest.mark.parametrize("external", ["tracked", "untracked", "staged", "head"])
async def test_external_changes_after_prior_integration_are_rejected(repository, external):
    worker = FakeWorker()
    async with WorkerCoordinator(repository, {"worker": worker}) as coordinator:
        first = (await coordinator.submit_worker_task("worker", "first"))["task_id"]
        second = (await coordinator.submit_worker_task("worker", "second"))["task_id"]
        trees = [tree for tree in worker.workspaces.values() if tree is not None]
        (trees[0] / "first.py").write_text("first\n")
        (trees[1] / "second.py").write_text("second\n")
        await approve_patch(coordinator, first)
        if external == "tracked":
            (repository / "hello.py").write_text("external user edit\n")
        elif external == "untracked":
            (repository / "external.py").write_text("external user edit\n")
        elif external == "staged":
            git(repository, "add", "first.py")
        else:
            git(repository, "commit", "--allow-empty", "-qm", "external head")
        with pytest.raises(WorkerError, match="prior approved changes"):
            await coordinator.review_worker_changes(second)
        assert (repository / "first.py").read_text() == "first\n"
        assert not (repository / "second.py").exists()


async def test_conflicting_approved_patch_preserves_prior_result(repository):
    worker = FakeWorker()
    async with WorkerCoordinator(repository, {"worker": worker}) as coordinator:
        first = (await coordinator.submit_worker_task("worker", "first"))["task_id"]
        second = (await coordinator.submit_worker_task("worker", "second"))["task_id"]
        trees = [tree for tree in worker.workspaces.values() if tree is not None]
        (trees[0] / "hello.py").write_text("first\n")
        (trees[1] / "hello.py").write_text("conflict\n")
        await approve_patch(coordinator, first)
        with pytest.raises(WorkerError, match="Git operation failed"):
            await approve_patch(coordinator, second)
        assert (repository / "hello.py").read_text() == "first\n"
        assert git(repository, "diff", "--cached") == ""


async def test_submodule_in_later_diff_entry_is_rejected(repository):
    worker = FakeWorker()
    async with WorkerCoordinator(repository, {"worker": worker}) as coordinator:
        task_id = (await coordinator.submit_worker_task("worker", "change"))["task_id"]
        tree = next(iter(worker.workspaces.values()))
        assert tree is not None
        (tree / "hello.py").write_text("first diff entry\n")
        nested = tree / "z-submodule"
        nested.mkdir()
        git(nested, "init", "-q")
        (nested / "nested.py").write_text("nested\n")
        git(nested, "add", ".")
        git(nested, "commit", "-qm", "nested repository")
        try:
            with pytest.raises(WorkerError, match="unsupported submodule"):
                await coordinator.review_worker_changes(task_id)
        finally:
            shutil.rmtree(nested)
        assert (repository / "hello.py").read_text() == "original\n"


async def test_binary_addition_and_deletion_apply_exactly(repository):
    worker = FakeWorker()
    async with WorkerCoordinator(repository, {"worker": worker}) as coordinator:
        task_id = (await coordinator.submit_worker_task("worker", "change"))["task_id"]
        tree = next(iter(worker.workspaces.values()))
        assert tree is not None
        binary = b"\x00\xff\x01\x02worker artifact"
        (tree / "artifact.bin").write_bytes(binary)
        (tree / "hello.py").unlink()
        review = await coordinator.review_worker_changes(task_id)
        assert "GIT binary patch" in review["diff"]
        assert (await approve_patch(coordinator, task_id))["result"]["state"] == "integrated"
        assert (repository / "artifact.bin").read_bytes() == binary
        assert not (repository / "hello.py").exists()
        assert git(repository, "diff", "--cached") == ""


async def test_failed_submission_cleanup_remains_retryable(repository, monkeypatch):
    coordinator = WorkerCoordinator(repository, {"worker": FakeWorker(fail=True)})
    original = coordinator._remove_worktree
    attempts = []

    def remove(worktree):
        if not attempts:
            attempts.append(worktree)
            msg = "Safe cleanup failure."
            raise WorkerError(msg)
        original(worktree)

    monkeypatch.setattr(coordinator, "_remove_worktree", remove)
    with pytest.raises(WorkerError, match="cleanup failure"):
        await coordinator.submit_worker_task("worker", "change")
    assert await coordinator.list_worker_tasks() == []
    assert git(repository, "worktree", "list", "--porcelain").count("worktree ") == 2
    await coordinator.aclose()
    assert git(repository, "worktree", "list", "--porcelain").count("worktree ") == 1


@pytest.mark.parametrize("configuration", ["diff.noprefix", "diff.mnemonicPrefix"])
async def test_canonical_diff_paths_ignore_repository_prefix_configuration(repository, configuration):
    git(repository, "config", configuration, "true")
    git(repository, "config", "diff.relative", "true")
    worker = FakeWorker()
    async with WorkerCoordinator(repository, {"worker": worker}) as coordinator:
        task_id = (await coordinator.submit_worker_task("worker", "change"))["task_id"]
        tree = next(iter(worker.workspaces.values()))
        assert tree is not None
        (tree / "nested").mkdir()
        (tree / "nested" / "artifact.py").write_text("approved nested content\n")
        review = await coordinator.review_worker_changes(task_id)
        assert "diff --git a/nested/artifact.py b/nested/artifact.py" in review["diff"]
        assert (await approve_patch(coordinator, task_id))["result"]["state"] == "integrated"
        assert (repository / "nested" / "artifact.py").read_text() == "approved nested content\n"
        assert not (repository / "artifact.py").exists()


async def test_approved_whitespace_is_not_rewritten_by_repository_configuration(repository):
    git(repository, "config", "apply.whitespace", "fix")
    worker = FakeWorker()
    async with WorkerCoordinator(repository, {"worker": worker}) as coordinator:
        task_id = (await coordinator.submit_worker_task("worker", "change"))["task_id"]
        tree = next(iter(worker.workspaces.values()))
        assert tree is not None
        approved = b"literal trailing spaces  \n\t\n"
        (tree / "hello.py").write_bytes(approved)
        assert (await approve_patch(coordinator, task_id))["result"]["state"] == "integrated"
        assert (repository / "hello.py").read_bytes() == approved
