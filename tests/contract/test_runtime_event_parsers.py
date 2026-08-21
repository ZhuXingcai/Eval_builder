from __future__ import annotations

from pathlib import Path

from env_mock_agent.runtimes.base import RuntimeEventSequencer
from env_mock_agent.runtimes.claude_cli import ClaudeCodeCliRuntime
from env_mock_agent.runtimes.pi_rpc import PiRpcRuntime
from env_mock_agent.schemas import (
    ModelProfile,
    RuntimeEventType,
    RuntimeName,
    RuntimeRequest,
)


def make_request(tmp_path: Path) -> RuntimeRequest:
    return RuntimeRequest(
        run_id="run",
        artifact_id="artifact",
        role="builder",
        workspace=str(tmp_path),
        prompt="Build an input.",
        model_profile=ModelProfile(provider="test", model="test"),
    )


def test_claude_cli_hook_events_normalize_to_tool_pair(tmp_path: Path) -> None:
    request = make_request(tmp_path)
    events = RuntimeEventSequencer(request, RuntimeName.CLAUDE_CODE_CLI, "handle")
    started = ClaudeCodeCliRuntime._normalize_event(
        events,
        {
            "type": "system",
            "subtype": "hook_started",
            "hook_event": "PreToolUse",
            "tool_name": "Write",
            "tool_use_id": "tool-1",
        },
    )
    finished = ClaudeCodeCliRuntime._normalize_event(
        events,
        {
            "type": "system",
            "subtype": "hook_response",
            "hook_event": "PostToolUse",
            "tool_name": "Write",
            "tool_use_id": "tool-1",
        },
    )
    assert started.event_type == RuntimeEventType.TOOL_STARTED
    assert finished.event_type == RuntimeEventType.TOOL_FINISHED
    assert started.tool_call_id == finished.tool_call_id == "tool-1"


def test_pi_bridge_events_normalize_to_tool_pair(tmp_path: Path) -> None:
    request = make_request(tmp_path)
    events = RuntimeEventSequencer(request, RuntimeName.PI_RPC, "handle")
    started = PiRpcRuntime._normalize(
        events,
        "tool_execution_start",
        {"toolName": "write", "toolCallId": "tool-1"},
    )
    finished = PiRpcRuntime._normalize(
        events,
        "tool_execution_end",
        {"toolName": "write", "toolCallId": "tool-1"},
    )
    assert started.event_type == RuntimeEventType.TOOL_STARTED
    assert finished.event_type == RuntimeEventType.TOOL_FINISHED
    assert started.tool_call_id == finished.tool_call_id == "tool-1"


def test_pi_ark_model_config_references_environment_without_writing_secret(
    tmp_path: Path,
) -> None:
    profile = ModelProfile(provider="ark", model="glm-5-2-260617")
    PiRpcRuntime._prepare_model_config(tmp_path, profile)
    content = (tmp_path / ".pi-home/models.json").read_text(encoding="utf-8")
    assert "$ARK_API_KEY" in content
    assert "glm-5-2-260617" in content
