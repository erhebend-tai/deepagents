"""Devin v1 sessions over a fixed, non-redirecting HTTPS transport."""

import asyncio
import json
import math
import os
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from deepagents.workers.types import WorkerError, WorkerResult, WorkerState, WorkerStatus

_ENDPOINT = "https://api.devin.ai/v1/sessions"
_MAX_RESPONSE = 1_048_576
_MAX_OUTPUT = 65_536
_ID = re.compile(r"[A-Za-z0-9_-]{1,200}", re.ASCII)
_REMOTE_INSTRUCTIONS = (
    "Work in your remote workspace on a separate task branch. Do not merge, "
    "push to the default branch, or modify another worker's branch. Return a "
    "summary, tests, and branch/PR references for human review. Your remote "
    "workspace is not mounted locally; returned text is never a local patch.\n\n"
)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args: object, **_kwargs: object) -> None:
        return None


def _session_id(value: object) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        msg = "Devin returned an invalid session identifier."
        raise WorkerError(msg)
    return value


class DevinWorker:
    """Opt-in Devin adapter; credentials are never included in diagnostics.

    !!! warning "Experimental"
        This adapter targets the Devin v1 sessions API. Remote sessions are not
        local worktrees and cannot be integrated by `WorkerCoordinator`.

    Args:
        api_key: Host-provided credential, otherwise read from `DEVIN_API_KEY`.
        timeout: Finite per-request socket timeout in seconds, at most 120.
    """

    is_local = False

    def __init__(self, *, api_key: str | None = None, timeout: float = 30) -> None:
        """Configure the fixed-endpoint transport without making a request."""
        if not math.isfinite(timeout) or not 0 < timeout <= 120:  # noqa: PLR2004  # Explicit request deadline ceiling.
            msg = "Devin timeout must be between 0 and 120 seconds."
            raise ValueError(msg)
        self._api_key = api_key if api_key is not None else os.environ.get("DEVIN_API_KEY", "")
        self._timeout = timeout
        self._sessions: set[str] = set()
        self._terminated: dict[str, tuple[WorkerState, dict[str, object]]] = {}
        self._closed = False

    def _request(self, method: str, session: str | None = None, *, prompt: str | None = None) -> dict[str, object]:
        if not self._api_key:
            msg = "Devin authentication is not configured."
            raise WorkerError(msg)
        url = _ENDPOINT if session is None else f"{_ENDPOINT}/{_session_id(session)}"
        data = json.dumps({"prompt": prompt}).encode() if prompt is not None else None
        request = Request(  # noqa: S310  # Fixed HTTPS origin plus strictly validated opaque session ID.
            url, data=data, headers={"Authorization": "Bearer " + self._api_key, "Content-Type": "application/json"}, method=method
        )
        try:
            # Ignore ambient proxies and never forward Authorization to a redirect.
            opener = build_opener(ProxyHandler({}), _NoRedirect())
            with opener.open(request, timeout=self._timeout) as response:
                raw = response.read(_MAX_RESPONSE + 1)
            if len(raw) > _MAX_RESPONSE:
                msg = "Devin response exceeded the size limit."
                raise WorkerError(msg)
            payload = json.loads(raw) if raw else {}
            if not isinstance(payload, dict):
                msg = "Devin returned an invalid response."
                raise WorkerError(msg)
        except HTTPError as exc:
            exc.close()
            msg = "Devin request failed."
            raise WorkerError(msg) from None
        except (URLError, OSError, ValueError):
            msg = "Devin request failed."
            raise WorkerError(msg) from None
        return payload

    def _known(self, handle: str) -> None:
        if handle not in self._sessions and handle not in self._terminated:
            msg = "Unknown Devin session."
            raise WorkerError(msg)

    async def submit(self, prompt: str, *, cwd: Path | None = None) -> str:
        """Create a remote session.

        Args:
            prompt: Instructions sent only to Devin.
            cwd: Must be `None`; remote workspaces are not local directories.
        """
        if self._closed or cwd is not None:
            msg = "Devin worker is closed or received a local workspace."
            raise WorkerError(msg)
        request = asyncio.create_task(asyncio.to_thread(self._request, "POST", prompt=_REMOTE_INSTRUCTIONS + prompt))
        try:
            payload = await asyncio.shield(request)
        except asyncio.CancelledError:
            payload = await request
            session = _session_id(payload.get("session_id"))
            self._sessions.add(session)
            await self.cancel(session)
            raise
        session = _session_id(payload.get("session_id"))
        self._sessions.add(session)
        return session

    async def _observe(self, handle: str) -> tuple[WorkerState, dict[str, object]]:
        self._known(handle)
        if handle in self._terminated:
            return self._terminated[handle]
        payload = await asyncio.to_thread(self._request, "GET", handle)
        state = {
            "finished": WorkerState.COMPLETED,
            "stopped": WorkerState.CANCELLED,
            "error": WorkerState.FAILED,
            "blocked": WorkerState.BLOCKED,
        }.get(str(payload.get("status_enum")), WorkerState.RUNNING)
        return state, payload

    async def status(self, handle: str) -> WorkerStatus:
        """Poll a session.

        Args:
            handle: Private session handle returned by `submit`.
        """
        state, _ = await self._observe(handle)
        return WorkerStatus(state)

    async def result(self, handle: str) -> WorkerResult:
        """Read bounded JSON/text output; error bodies are never returned.

        Args:
            handle: Private session handle returned by `submit`.
        """
        state, payload = await self._observe(handle)
        if state is not WorkerState.COMPLETED:
            return WorkerResult(state)
        output = json.dumps(payload.get("messages", []), ensure_ascii=True).replace(self._api_key, "[redacted]")[:_MAX_OUTPUT]
        structured = json.dumps(payload.get("structured_output"), ensure_ascii=True).replace(self._api_key, "[redacted]")
        if len(structured) > _MAX_OUTPUT:
            structured = '{"truncated": true}'
        return WorkerResult(state, output, structured)

    async def cancel(self, handle: str) -> None:
        """Request cancellation and confirm terminal state before retiring it.

        Args:
            handle: Private session handle returned by `submit`.
        """
        self._known(handle)
        if handle in self._terminated:
            return
        await asyncio.to_thread(self._request, "DELETE", handle)
        state, payload = await self._observe(handle)
        if state.terminal:
            self._sessions.discard(handle)
            self._terminated[handle] = state, payload

    async def aclose(self) -> None:
        """Terminate all owned sessions; failed termination remains retryable."""
        self._closed = True
        failed = False
        for session in tuple(self._sessions):
            try:
                await self.cancel(session)
            except WorkerError:
                failed = True
        if failed or self._sessions:
            msg = "Some Devin sessions could not be terminated; retry aclose()."
            raise WorkerError(msg)
