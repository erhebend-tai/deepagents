"""CLI argument, output, timeout, and process-group lifetime tests."""

import asyncio
import os
import signal
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest

from deepagents.workers import CopilotWorker, WorkerError, WorkerState


async def completed(worker, handle):
    async with asyncio.timeout(5):
        while not (status := await worker.status(handle)).state.terminal:  # noqa: ASYNC110  # Poll the public API, which has no completion event.
            await asyncio.sleep(0.01)
    return status.state


@pytest.mark.parametrize("credential", ["test-token", "short"])
async def test_argument_vector_environment_and_bounded_output(monkeypatch, workspace, credential):
    monkeypatch.setenv("DEVIN_API_KEY", "devin-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", credential)
    monkeypatch.setenv("COPILOT_ALLOW_ALL", "true")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "12")
    stream = asyncio.StreamReader()
    stream.feed_data(b"x" * 100_000 + credential.encode())
    stream.feed_eof()
    process = Mock(stdout=stream, pid=12345, returncode=0, wait=AsyncMock(return_value=0))
    launch = AsyncMock(return_value=process)
    kill = Mock()
    monkeypatch.setattr("deepagents.workers.copilot.shutil.which", lambda _executable: "/usr/bin/copilot")
    monkeypatch.setattr("deepagents.workers.copilot.asyncio.create_subprocess_exec", launch)
    monkeypatch.setattr("deepagents.workers.copilot.os.killpg", kill)
    prompt = "task; $(touch should-not-exist)\n--allow-all-tools"
    worker = CopilotWorker()
    handle = await worker.submit(prompt, cwd=workspace)
    assert await completed(worker, handle) is WorkerState.COMPLETED
    arguments = launch.call_args.args
    assert arguments == ("/usr/bin/copilot", "-p", prompt, "--deny-tool", "shell", "--allow-tool", "read", "--allow-tool", "write")
    environment = launch.call_args.kwargs["env"]
    assert environment["COPILOT_GITHUB_TOKEN"] == credential
    assert "DEVIN_API_KEY" not in environment
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert "GIT_CONFIG_COUNT" not in environment
    assert "COPILOT_ALLOW_ALL" not in environment
    assert launch.call_args.kwargs["cwd"] == workspace
    assert launch.call_args.kwargs["start_new_session"] is (os.name == "posix")
    result = await worker.result(handle)
    assert len(result.output) <= 65_536
    assert credential not in result.output
    assert "[redacted]" in result.output
    assert credential not in repr(worker)
    await worker.aclose()
    if os.name == "posix":
        kill.assert_called_with(process.pid, signal.SIGKILL)


async def test_missing_or_unstartable_cli_is_safe(monkeypatch, workspace):
    worker = CopilotWorker(executable="missing")
    monkeypatch.setattr("deepagents.workers.copilot.shutil.which", lambda _executable: None)
    with pytest.raises(WorkerError, match="not installed"):
        await worker.submit("sensitive prompt", cwd=workspace)
    monkeypatch.setattr("deepagents.workers.copilot.shutil.which", lambda _executable: "/bad/path")
    monkeypatch.setattr("deepagents.workers.copilot.asyncio.create_subprocess_exec", AsyncMock(side_effect=OSError("sensitive prompt")))
    with pytest.raises(WorkerError) as error:
        await worker.submit("sensitive prompt", cwd=workspace)
    assert str(error.value) == "Copilot could not be started."
    await worker.aclose()


def executable(workspace, body):
    script = workspace / "fake-copilot"
    script.write_text(f"#!{sys.executable}\n{body}\n")
    script.chmod(0o700)
    return str(script)


async def test_real_process_high_volume_output_and_failure(workspace):
    program = executable(workspace, "import sys\nsys.stdout.write('x' * 500000)\nsys.stdout.flush()\nsys.exit(3)")
    async with _closing(CopilotWorker(executable=program)) as worker:
        handle = await worker.submit("task", cwd=workspace)
        assert await completed(worker, handle) is WorkerState.FAILED
        assert len((await worker.result(handle)).output) == 65_536


@asynccontextmanager
async def _closing(worker: CopilotWorker) -> AsyncIterator[CopilotWorker]:
    try:
        yield worker
    finally:
        await worker.aclose()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
@pytest.mark.parametrize("operation", ["timeout", "cancel", "close"])
async def test_deadline_and_cancellation_kill_children(workspace, operation):
    program = executable(
        workspace,
        (
            "import os, pathlib, subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "pathlib.Path('pids').write_text(str(os.getpid()) + ' ' + str(child.pid))\n"
            "print('started', flush=True)\n"
            "time.sleep(60)"
        ),
    )
    worker = CopilotWorker(executable=program, timeout=0.3 if operation == "timeout" else 30)
    try:
        handle = await worker.submit("task", cwd=workspace)
        async with asyncio.timeout(3):
            while not (workspace / "pids").exists():  # noqa: ASYNC110  # Wait for a real subprocess's file, not an in-process event.
                await asyncio.sleep(0.01)
        parent, child = map(int, (workspace / "pids").read_text().split())
        if operation == "cancel":
            await worker.cancel(handle)
        elif operation == "close":
            await worker.aclose()
        expected = WorkerState.TIMED_OUT if operation == "timeout" else WorkerState.CANCELLED
        assert await completed(worker, handle) is expected
        with pytest.raises(ProcessLookupError):
            os.kill(parent, 0)
        # An orphan may briefly be a zombie pending reaping by the host's init;
        # it must never remain executing.
        proc = workspace.__class__(f"/proc/{child}/stat")
        assert not proc.exists() or proc.read_text().split()[2] == "Z"
    finally:
        await worker.aclose()


async def test_immediate_cancellation_does_not_leave_monitor_or_pipe(workspace):
    program = executable(workspace, "import time\nprint('started', flush=True)\ntime.sleep(60)")
    worker = CopilotWorker(executable=program)
    handle = await worker.submit("task", cwd=workspace)
    await asyncio.wait_for(worker.cancel(handle), 3)
    assert (await worker.status(handle)).state is WorkerState.CANCELLED
    await worker.aclose()


@pytest.mark.parametrize("permissions", [("*",), ("shell",), ("shell(*)",), ("--allow-all-tools",)])
def test_no_broad_tool_permissions(permissions):
    with pytest.raises(ValueError, match="permissions"):
        CopilotWorker(allowed_tools=permissions)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_deadline(timeout):
    with pytest.raises(ValueError, match="timeout"):
        CopilotWorker(timeout=timeout)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group ID reuse")
@pytest.mark.parametrize("outcome", ["completed", "timeout"])
async def test_terminal_cleanup_never_signals_historic_process_group_again(monkeypatch, workspace, outcome):
    stream = asyncio.StreamReader()
    stream.feed_eof()
    kill = Mock()

    async def wait():
        if outcome == "timeout" and not kill.called:
            await asyncio.Event().wait()
        return 0 if outcome == "completed" else -9

    process = Mock(stdout=stream, pid=43210, returncode=None, wait=AsyncMock(side_effect=wait))
    monkeypatch.setattr("deepagents.workers.copilot.shutil.which", lambda _executable: "/usr/bin/copilot")
    monkeypatch.setattr("deepagents.workers.copilot.asyncio.create_subprocess_exec", AsyncMock(return_value=process))
    monkeypatch.setattr("deepagents.workers.copilot.os.killpg", kill)
    worker = CopilotWorker(timeout=0.01 if outcome == "timeout" else 1)
    handle = await worker.submit("task", cwd=workspace)
    expected = WorkerState.COMPLETED if outcome == "completed" else WorkerState.TIMED_OUT
    assert await completed(worker, handle) is expected
    assert kill.call_count == 1
    await worker.cancel(handle)
    await worker.cancel(handle)
    await worker.aclose()
    await worker.aclose()
    assert kill.call_count == 1
