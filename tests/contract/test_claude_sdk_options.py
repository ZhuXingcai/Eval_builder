from __future__ import annotations

from pathlib import Path

import pytest

from env_mock_agent.runtimes import ClaudeAgentSdkRuntime
from env_mock_agent.schemas import (
    ModelProfile,
    RuntimeName,
    RuntimeRequest,
    RuntimeSessionRef,
)


def make_request(workspace: Path) -> RuntimeRequest:
    return RuntimeRequest(
        run_id="run",
        artifact_id="artifact",
        role="builder",
        workspace=str(workspace),
        prompt="Create an input.",
        model_profile=ModelProfile(provider="anthropic", model="test"),
        max_turns=7,
        max_budget_usd=1.5,
    )


def test_sdk_options_preserve_budget_and_resume_session(tmp_path: Path) -> None:
    request = make_request(tmp_path)
    session = RuntimeSessionRef(
        runtime=RuntimeName.CLAUDE_AGENT_SDK,
        run_handle="handle",
        external_session_id="session-1",
        workspace=str(tmp_path),
        model_profile=request.model_profile,
    )
    options = ClaudeAgentSdkRuntime()._options(request, tmp_path, session)
    assert options.resume == "session-1"
    assert options.max_turns == 7
    assert options.max_budget_usd == 1.5
    assert options.setting_sources == []


@pytest.mark.asyncio
async def test_sdk_pretool_hook_denies_path_escape(tmp_path: Path) -> None:
    request = make_request(tmp_path)
    options = ClaudeAgentSdkRuntime()._options(request, tmp_path, None)
    assert options.hooks is not None
    matcher = options.hooks["PreToolUse"][0]
    callback = matcher.hooks[0]
    result = await callback(
        {"tool_name": "Write", "tool_input": {"file_path": "../outside.txt"}},
        "tool-1",
        {"signal": None},
    )
    decision = result["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
