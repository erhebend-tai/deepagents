"""Devin sessions transport, state mapping, and credential safety."""

import asyncio
import io
import json
import threading
from email.message import Message
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest

from deepagents.workers import DevinWorker, WorkerError, WorkerState
from deepagents.workers.devin import _NoRedirect


def transport(monkeypatch, responses):
    requests = []
    queue = iter(responses)

    def open_request(request, *, timeout):
        requests.append((request, timeout))
        value = next(queue)
        if isinstance(value, Exception):
            raise value
        return io.BytesIO(json.dumps(value).encode())

    opener = Mock(open=open_request)
    factory = Mock(return_value=opener)
    monkeypatch.setattr("deepagents.workers.devin.build_opener", factory)
    return requests, factory


async def test_fixed_api_create_poll_result_and_terminate(monkeypatch):
    requests, factory = transport(
        monkeypatch,
        [
            {"session_id": "devin_abc-123"},
            {"status_enum": "working"},
            {"status_enum": "finished", "messages": [{"message": "done"}], "structured_output": {"tests": "passed"}},
            {"status": "running", "status_enum": "finished"},
            {},
            {"status": "exit", "status_enum": "stopped"},
        ],
    )
    worker = DevinWorker(api_key="private-test-credential", timeout=7)
    handle = await worker.submit("implement feature")
    assert (await worker.status(handle)).state is WorkerState.RUNNING
    result = await worker.result(handle)
    assert result.state is WorkerState.COMPLETED
    assert result.structured_output is not None
    assert json.loads(result.structured_output) == {"tests": "passed"}
    assert "done" in result.output
    await worker.cancel(handle)
    await worker.aclose()
    assert [request.get_method() for request, _ in requests] == ["POST", "GET", "GET", "GET", "DELETE", "GET"]
    assert [request.full_url for request, _ in requests] == [
        "https://api.devin.ai/v1/sessions",
        *["https://api.devin.ai/v1/sessions/devin_abc-123"] * 5,
    ]
    assert all(timeout == 7 for _, timeout in requests)
    request = requests[0][0]
    assert request.get_header("Authorization") == "Bearer " + "private-test-credential"
    assert "separate task branch" in json.loads(request.data)["prompt"]
    assert json.loads(request.data)["prompt"].endswith("implement feature")
    handlers = factory.call_args.args
    assert handlers[0].proxies == {}
    assert isinstance(handlers[1], _NoRedirect)
    assert handlers[1].redirect_request(request, None, 302, "redirect", {}, "https://attacker.invalid") is None
    assert "private-test-credential" not in repr(worker)


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("finished", WorkerState.COMPLETED),
        ("stopped", WorkerState.CANCELLED),
        ("error", WorkerState.FAILED),
        ("blocked", WorkerState.BLOCKED),
        ("working", WorkerState.RUNNING),
        ("complete", WorkerState.RUNNING),
        ("unknown-new-state", WorkerState.RUNNING),
        (None, WorkerState.RUNNING),
    ],
)
async def test_provider_states_fail_closed(monkeypatch, provider, expected):
    transport(
        monkeypatch,
        [
            {"session_id": "abc"},
            {"status_enum": provider},
            {"status": "running", "status_enum": provider},
            {},
            {"status": "exit", "status_enum": "stopped"},
        ],
    )
    worker = DevinWorker(api_key="credential")
    handle = await worker.submit("task")
    assert (await worker.status(handle)).state is expected
    assert expected.terminal is (provider in {"finished", "stopped", "error"})
    await worker.aclose()


@pytest.mark.parametrize("identifier", ["../secret", "https://attacker.invalid", "x?key=y", "x/y", "x%2fy", "", "a\nb", 42, "a" * 201])
async def test_session_path_injection_is_rejected(monkeypatch, identifier):
    requests, _ = transport(monkeypatch, [{"session_id": identifier}])
    worker = DevinWorker(api_key="credential")
    with pytest.raises(WorkerError, match="invalid session identifier"):
        await worker.submit("task")
    assert len(requests) == 1
    with pytest.raises(WorkerError, match="Unknown"):
        await worker.status("../secret")
    assert len(requests) == 1


async def test_missing_auth_never_opens_transport(monkeypatch):
    monkeypatch.delenv("DEVIN_API_KEY", raising=False)
    requests, _ = transport(monkeypatch, [])
    worker = DevinWorker()
    with pytest.raises(WorkerError, match="authentication"):
        await worker.submit("payload")
    assert not requests


@pytest.mark.parametrize(
    "failure",
    [
        HTTPError("https://api.devin.ai/v1/sessions", 401, "credential and payload", Message(), io.BytesIO(b"sensitive error body")),
        URLError("credential and payload"),
        TimeoutError("credential and payload"),
    ],
)
async def test_transport_errors_are_sanitized(monkeypatch, failure):
    transport(monkeypatch, [failure])
    worker = DevinWorker(api_key="credential")
    with pytest.raises(WorkerError) as error:
        await worker.submit("payload")
    assert str(error.value) == "Devin request failed."
    assert error.value.__suppress_context__
    assert "credential" not in repr(error.value)
    assert "payload" not in repr(error.value)


async def test_failed_session_does_not_return_error_body(monkeypatch):
    transport(
        monkeypatch,
        [
            {"session_id": "abc"},
            {"status_enum": "error", "messages": ["private payload"], "structured_output": {"error": "secret"}},
            {"status": "running", "status_enum": "error"},
            {},
            {"status": "exit", "status_enum": "stopped"},
        ],
    )
    worker = DevinWorker(api_key="credential")
    handle = await worker.submit("task")
    result = await worker.result(handle)
    assert result.output == ""
    assert result.structured_output is None
    await worker.aclose()


async def test_response_bounds_redaction_and_valid_structured_json(monkeypatch):
    transport(
        monkeypatch,
        [
            {"session_id": "abc"},
            {"status_enum": "finished", "messages": ["credential"], "structured_output": {"large": "x" * 70_000}},
            {"status": "running", "status_enum": "finished"},
            {},
            {"status": "exit", "status_enum": "stopped"},
        ],
    )
    worker = DevinWorker(api_key="credential")
    handle = await worker.submit("task")
    result = await worker.result(handle)
    assert "credential" not in result.output
    assert result.structured_output is not None
    assert json.loads(result.structured_output) == {"truncated": True}
    await worker.aclose()


async def test_close_retries_remote_termination(monkeypatch):
    requests, _ = transport(
        monkeypatch,
        [
            {"session_id": "abc"},
            {"status": "running"},
            URLError("private"),
            {"status": "running"},
            {},
            {"status": "exit", "status_enum": "stopped"},
        ],
    )
    worker = DevinWorker(api_key="credential")
    await worker.submit("task")
    with pytest.raises(WorkerError, match="retry aclose"):
        await worker.aclose()
    await worker.aclose()
    assert [request.get_method() for request, _ in requests] == ["POST", "GET", "DELETE", "GET", "DELETE", "GET"]
    with pytest.raises(WorkerError, match="closed"):
        await worker.submit("another task")


async def test_cancelled_submission_terminates_created_session(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    methods = []

    def request(method, _session=None, **_kwargs: object):
        methods.append(method)
        if method == "POST":
            started.set()
            assert release.wait(3)
            return {"session_id": "abc"}
        if method == "GET":
            return {"status": "exit", "status_enum": "stopped"} if "DELETE" in methods else {"status": "running"}
        return {}

    worker = DevinWorker(api_key="credential")
    monkeypatch.setattr(worker, "_request", request)
    submission = asyncio.create_task(worker.submit("task"))
    assert await asyncio.to_thread(started.wait, 3)
    submission.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await submission
    assert methods == ["POST", "GET", "DELETE", "GET"]
    await worker.aclose()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), 121])
def test_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="timeout"):
        DevinWorker(timeout=timeout)


async def test_delete_acknowledgement_is_not_terminal_confirmation(monkeypatch):
    transport(
        monkeypatch,
        [
            {"session_id": "abc"},
            {"status": "running", "status_enum": "working"},
            {},
            {"status_enum": "unknown-new-state"},
            {"status_enum": "working"},
            {"status": "running", "status_enum": "working"},
            {},
            {"status_enum": "blocked"},
            {"status": "running", "status_enum": "working"},
            {},
            {"status": "exit", "status_enum": "stopped"},
        ],
    )
    worker = DevinWorker(api_key="credential")
    handle = await worker.submit("task")
    await worker.cancel(handle)
    assert (await worker.status(handle)).state is WorkerState.RUNNING
    with pytest.raises(WorkerError, match="retry aclose"):
        await worker.aclose()
    await worker.aclose()
    assert (await worker.status(handle)).state is WorkerState.CANCELLED


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("finished", WorkerState.COMPLETED),
        ("error", WorkerState.FAILED),
        ("stopped", WorkerState.CANCELLED),
        ("unknown", WorkerState.RUNNING),
    ],
)
@pytest.mark.parametrize("observe_first", [True, False])
async def test_already_exited_sessions_never_receive_delete(monkeypatch, provider, expected, observe_first):
    requests, _ = transport(
        monkeypatch,
        [
            {"session_id": "abc"},
            {"status": "exit", "status_enum": provider},
        ],
    )
    worker = DevinWorker(api_key="credential")
    handle = await worker.submit("task")
    if observe_first:
        assert (await worker.status(handle)).state is expected
    await worker.cancel(handle)
    await worker.aclose()
    assert (await worker.status(handle)).state is expected
    assert [request.get_method() for request, _ in requests] == ["POST", "GET"]


async def test_finished_task_in_running_session_still_requires_delete(monkeypatch):
    requests, _ = transport(
        monkeypatch,
        [
            {"session_id": "abc"},
            {"status": "running", "status_enum": "finished"},
            {},
            {"status": "exit", "status_enum": "stopped"},
        ],
    )
    worker = DevinWorker(api_key="credential")
    handle = await worker.submit("task")
    await worker.cancel(handle)
    await worker.aclose()
    assert [request.get_method() for request, _ in requests] == ["POST", "GET", "DELETE", "GET"]
