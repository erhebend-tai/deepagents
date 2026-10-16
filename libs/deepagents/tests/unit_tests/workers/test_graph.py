"""Worker tools and advisory subagents run through the existing SDK graph."""

import json

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from deepagents import SubAgent, create_deep_agent
from deepagents.workers import WorkerCoordinator
from tests.unit_tests.chat_model import GenericFakeChatModel
from tests.unit_tests.workers.test_coordinator import FakeWorker


def advisor(name: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {"file_path": "/advisory-notes.txt", "content": f"{name} private notes"},
                            "id": f"{name}-notes",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content=f"{name} recommendation"),
            ]
        )
    )


async def test_sdk_delegates_to_both_advisors_and_external_worker(repository):
    reviewer = advisor("bedrock")
    planner = advisor("ollama")
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {"description": "Review changes", "subagent_type": "bedrock-reviewer"},
                            "id": "review",
                            "type": "tool_call",
                        },
                        {"name": "task", "args": {"description": "Plan tasks", "subagent_type": "ollama-planner"}, "id": "plan", "type": "tool_call"},
                        {
                            "name": "submit_worker_task",
                            "args": {"worker": "copilot", "prompt": "Implement the plan"},
                            "id": "worker",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(content="Reconciled both recommendations and delegated coding."),
            ]
        )
    )
    subagents: list[SubAgent] = [
        {"name": "bedrock-reviewer", "description": "Advisory review", "model": reviewer},
        {"name": "ollama-planner", "description": "Advisory planning", "model": planner},
    ]
    worker = FakeWorker()
    async with WorkerCoordinator(repository, {"copilot": worker}) as coordinator:
        agent = create_deep_agent(model, tools=coordinator.tools(), subagents=subagents, checkpointer=InMemorySaver())
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Coordinate review, planning, and coding."}]},
            {"configurable": {"thread_id": "advisory-worker-wiring"}},
        )
        messages = {message.tool_call_id: message for message in result["messages"] if isinstance(message, ToolMessage)}
        assert "bedrock recommendation" in messages["review"].content
        assert "ollama recommendation" in messages["plan"].content
        assert "error" not in str(messages["review"].content).lower()
        assert "error" not in str(messages["plan"].content).lower()
        assert isinstance(messages["worker"].content, str)
        submitted = json.loads(messages["worker"].content)
        assert (await coordinator.check_worker_status(submitted["task_id"]))["state"] == "completed"
        assert len(worker.workspaces) == 1
        assert "/advisory-notes.txt" in result.get("files", {})
        assert not (repository / "advisory-notes.txt").exists()
        assert (repository / "hello.py").read_text() == "original\n"
        for chat_model in (model, reviewer, planner):
            assert all(not isinstance(tool, BaseTool) or tool.name != "execute" for tool in chat_model.tools)
        assert reviewer.call_history
        assert planner.call_history
        reviewer_tasks = [message.content for message in reviewer.call_history[0]["messages"] if message.type == "human"]
        planner_tasks = [message.content for message in planner.call_history[0]["messages"] if message.type == "human"]
        assert reviewer_tasks == ["Review changes"]
        assert planner_tasks == ["Plan tasks"]
