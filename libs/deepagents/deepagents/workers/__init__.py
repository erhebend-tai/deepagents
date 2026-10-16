"""Opt-in external workers; existing Deep Agents defaults are unchanged.

Pass `WorkerCoordinator.tools()` explicitly to an agent. Use one coordinator
per live session and always close it. Local worktrees are NOT security
sandboxes, and remote results are never executable local patches. Integration
requires a checkpointed LangGraph interrupt and host approval of an exact diff.
"""

from deepagents.workers.coordinator import WorkerCoordinator
from deepagents.workers.copilot import CopilotWorker
from deepagents.workers.devin import DevinWorker
from deepagents.workers.types import WorkerAdapter, WorkerError, WorkerResult, WorkerState, WorkerStatus

__all__ = ["CopilotWorker", "DevinWorker", "WorkerAdapter", "WorkerCoordinator", "WorkerError", "WorkerResult", "WorkerState", "WorkerStatus"]
