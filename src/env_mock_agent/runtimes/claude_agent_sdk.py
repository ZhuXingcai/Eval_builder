from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal, cast

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookContext,
    HookEventMessage,
    HookInput,
    HookJSONOutput,
    HookMatcher,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from env_mock_agent.runtimes.base import AgentRuntime, RuntimeEventSequencer
from env_mock_agent.runtimes.process import isolated_environment
from env_mock_agent.runtimes.security import validate_tool_input
from env_mock_agent.runtimes.workspace_manifest import changed_files, snapshot_workspace
from env_mock_agent.schemas import (
    RuntimeCapabilities,
    RuntimeErrorCode,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeName,
    RuntimeRequest,
    RuntimeResumeRequest,
    RuntimeSessionRef,
    RuntimeUsage,
)


def _usage_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float | str):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return 0


class ClaudeAgentSdkRuntime(AgentRuntime):
    def __init__(self) -> None:
        self._clients: dict[str, ClaudeSDKClient] = {}
        self._cancelled: set[str] = set()

    async def probe(self) -> RuntimeCapabilities:
        try:
            version = importlib.metadata.version("claude-agent-sdk")
        except importlib.metadata.PackageNotFoundError:
            return RuntimeCapabilities(
                name=RuntimeName.CLAUDE_AGENT_SDK,
                available=False,
                reason="claude-agent-sdk is not installed",
            )
        return RuntimeCapabilities(
            name=RuntimeName.CLAUDE_AGENT_SDK,
            available=True,
            version=version,
            tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"],
            supports_resume=True,
            reason="credential readiness is verified by live smoke",
        )

    async def run(self, request: RuntimeRequest) -> AsyncIterator[RuntimeEvent]:
        async for event in self._execute(request):
            yield event

    async def resume(self, resume_request: RuntimeResumeRequest) -> AsyncIterator[RuntimeEvent]:
        if not resume_request.session.external_session_id:
            raise ValueError("Claude Agent SDK resume requires external_session_id")
        request = resume_request.request.model_copy(update={"prompt": resume_request.prompt})
        async for event in self._execute(request, resume_request.session):
            yield event

    async def _execute(
        self,
        request: RuntimeRequest,
        resume_session: RuntimeSessionRef | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        workspace = Path(request.workspace).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        handle = f"claude-sdk:{request.run_id}:{request.artifact_id}"
        events = RuntimeEventSequencer(request, RuntimeName.CLAUDE_AGENT_SDK, handle)
        before = snapshot_workspace(workspace)
        options = self._options(request, workspace, resume_session)
        client = ClaudeSDKClient(options)
        self._clients[handle] = client
        yield events.event(RuntimeEventType.RUNTIME_STARTED, session=resume_session)

        session = resume_session
        usage: RuntimeUsage | None = None
        result_message = ""
        failure_message = ""
        failure_code: RuntimeErrorCode | None = None
        tool_calls = 0
        started_tools: set[str] = set()
        finished_tools: set[str] = set()
        try:
            async with asyncio.timeout(request.timeout_seconds):
                async with client:
                    await client.query(request.prompt)
                    async for message in client.receive_response():
                        if isinstance(message, HookEventMessage):
                            tool_event = self._hook_event(
                                events,
                                message,
                                started_tools,
                                finished_tools,
                            )
                            if tool_event is not None:
                                yield tool_event
                            continue
                        if isinstance(message, SystemMessage):
                            if message.subtype == "init":
                                external_session_id = str(message.data.get("session_id") or "")
                                if external_session_id:
                                    session = self._session(
                                        request,
                                        workspace,
                                        handle,
                                        external_session_id,
                                    )
                            elif (
                                message.subtype == "api_retry"
                                and str(message.data.get("error") or "") == "authentication_failed"
                            ):
                                failure_message = "Claude Agent SDK authentication failed"
                                failure_code = RuntimeErrorCode.AUTH_UNAVAILABLE
                                await client.interrupt()
                                break
                            continue
                        if isinstance(message, AssistantMessage):
                            if message.session_id:
                                session = self._session(
                                    request,
                                    workspace,
                                    handle,
                                    message.session_id,
                                )
                            for block in message.content:
                                if isinstance(block, TextBlock):
                                    yield events.event(
                                        RuntimeEventType.MESSAGE_DELTA,
                                        message=block.text,
                                        raw_type="assistant_text",
                                    )
                                elif isinstance(block, ToolUseBlock):
                                    if block.id in started_tools:
                                        continue
                                    started_tools.add(block.id)
                                    tool_calls += 1
                                    if tool_calls > request.max_tool_events:
                                        raise RuntimeError(
                                            f"Claude Agent SDK exceeded {request.max_tool_events} tool calls"
                                        )
                                    yield events.event(
                                        RuntimeEventType.TOOL_STARTED,
                                        message=block.name,
                                        tool_name=block.name,
                                        tool_call_id=block.id,
                                        raw_type="assistant_tool_use",
                                        payload={"input": block.input},
                                    )
                            continue
                        if isinstance(message, UserMessage) and isinstance(message.content, list):
                            for block in message.content:
                                if not isinstance(block, ToolResultBlock):
                                    continue
                                if block.tool_use_id in finished_tools:
                                    continue
                                finished_tools.add(block.tool_use_id)
                                yield events.event(
                                    RuntimeEventType.TOOL_FINISHED,
                                    tool_call_id=block.tool_use_id,
                                    raw_type="tool_result",
                                    payload={"is_error": bool(block.is_error)},
                                )
                            continue
                        if isinstance(message, ResultMessage):
                            if message.session_id:
                                session = self._session(
                                    request,
                                    workspace,
                                    handle,
                                    message.session_id,
                                )
                            usage = self._usage(message, tool_calls)
                            result_message = message.result or ""
                            if message.is_error:
                                failure_message = (
                                    "; ".join(message.errors or [])
                                    or result_message
                                    or "Claude Agent SDK failed"
                                )
                                failure_code = self._error_code(failure_message)
        except TimeoutError:
            with contextlib.suppress(Exception):
                await client.interrupt()
            failure_message = f"Claude Agent SDK timed out after {request.timeout_seconds}s"
            failure_code = RuntimeErrorCode.TIMEOUT
        except Exception as exc:
            with contextlib.suppress(Exception):
                await client.interrupt()
            failure_message = str(exc)
            failure_code = RuntimeErrorCode.PROCESS_ERROR
        finally:
            self._clients.pop(handle, None)

        if handle in self._cancelled:
            self._cancelled.discard(handle)
            failure_message = "Claude Agent SDK cancelled"
            failure_code = RuntimeErrorCode.CANCELLED
        for entry in changed_files(before, snapshot_workspace(workspace)):
            yield events.event(
                RuntimeEventType.ARTIFACT_WRITTEN,
                message=entry.relative_path,
                payload=entry.as_payload(),
            )
        yield events.event(
            RuntimeEventType.USAGE,
            usage=usage or RuntimeUsage(tool_calls=tool_calls),
            session=session,
        )
        if failure_message:
            yield events.event(
                RuntimeEventType.RUNTIME_FAILED,
                message=failure_message,
                session=session,
                error_code=failure_code,
            )
        else:
            yield events.event(
                RuntimeEventType.RUNTIME_FINISHED,
                message=result_message,
                session=session,
            )

    def _options(
        self,
        request: RuntimeRequest,
        workspace: Path,
        resume_session: RuntimeSessionRef | None,
    ) -> ClaudeAgentOptions:
        async def pre_tool_use(
            input_data: HookInput,
            _tool_use_id: str | None,
            _context: HookContext,
        ) -> HookJSONOutput:
            tool_name = str(input_data.get("tool_name") or "")
            tool_input = input_data.get("tool_input")
            reason = validate_tool_input(
                workspace,
                tool_name,
                tool_input if isinstance(tool_input, dict) else {},
            )
            if reason:
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    }
                }
            return {}

        tools = request.allowed_tools or ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
        effort = cast(
            Literal["low", "medium", "high", "xhigh", "max"],
            request.model_profile.effort or "high",
        )
        return ClaudeAgentOptions(
            tools=tools,
            allowed_tools=tools,
            disallowed_tools=request.disallowed_tools,
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
                "append": request.system_policy,
            },
            strict_mcp_config=True,
            permission_mode="acceptEdits",
            cwd=workspace,
            env=isolated_environment(workspace),
            max_turns=request.max_turns,
            max_budget_usd=request.max_budget_usd,
            model=request.model_profile.model,
            effort=effort,
            thinking={"type": "adaptive"},
            setting_sources=[],
            resume=resume_session.external_session_id if resume_session else None,
            hooks={
                "PreToolUse": [
                    HookMatcher(
                        matcher="Read|Write|Edit|Bash|Glob|Grep",
                        hooks=[pre_tool_use],
                    )
                ]
            },
            include_hook_events=True,
        )

    @staticmethod
    def _session(
        request: RuntimeRequest,
        workspace: Path,
        handle: str,
        external_session_id: str,
    ) -> RuntimeSessionRef:
        return RuntimeSessionRef(
            runtime=RuntimeName.CLAUDE_AGENT_SDK,
            run_handle=handle,
            external_session_id=external_session_id,
            session_path=str(workspace / ".runtime-home"),
            workspace=str(workspace),
            model_profile=request.model_profile,
        )

    @staticmethod
    def _hook_event(
        events: RuntimeEventSequencer,
        message: HookEventMessage,
        started_tools: set[str],
        finished_tools: set[str],
    ) -> RuntimeEvent | None:
        if message.hook_event_name not in {"PostToolUse", "PostToolUseFailure"}:
            return None
        tool_call_id = str(
            message.data.get("tool_use_id") or message.data.get("tool_call_id") or message.uuid or ""
        )
        if not tool_call_id or tool_call_id in finished_tools:
            return None
        finished_tools.add(tool_call_id)
        return events.event(
            RuntimeEventType.TOOL_FINISHED,
            tool_name=str(message.data.get("tool_name") or "") or None,
            tool_call_id=tool_call_id,
            raw_type=f"hook:{message.hook_event_name}:{message.subtype}",
            payload={str(key): value for key, value in message.data.items()},
        )

    @staticmethod
    def _usage(message: ResultMessage, tool_calls: int) -> RuntimeUsage:
        raw = message.usage or {}
        return RuntimeUsage(
            input_tokens=_usage_int(raw.get("input_tokens")),
            output_tokens=_usage_int(raw.get("output_tokens")),
            cache_creation_input_tokens=_usage_int(raw.get("cache_creation_input_tokens")),
            cache_read_input_tokens=_usage_int(raw.get("cache_read_input_tokens")),
            cost_usd=(float(message.total_cost_usd) if message.total_cost_usd is not None else None),
            tool_calls=tool_calls,
            turns=message.num_turns,
            duration_ms=message.duration_ms,
        )

    @staticmethod
    def _error_code(message: str) -> RuntimeErrorCode:
        lowered = message.lower()
        if "auth" in lowered or "api key" in lowered:
            return RuntimeErrorCode.AUTH_UNAVAILABLE
        if "budget" in lowered:
            return RuntimeErrorCode.BUDGET_EXCEEDED
        if "model" in lowered and ("not found" in lowered or "unavailable" in lowered):
            return RuntimeErrorCode.MODEL_UNAVAILABLE
        return RuntimeErrorCode.PROCESS_ERROR

    async def cancel(self, run_handle: str) -> None:
        self._cancelled.add(run_handle)
        client = self._clients.get(run_handle)
        if client is not None:
            await client.interrupt()
