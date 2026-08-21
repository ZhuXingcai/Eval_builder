from __future__ import annotations

from pydantic import BaseModel, Field

from env_mock_agent.runtimes.base import AgentRuntime
from env_mock_agent.runtimes.registry import RuntimeRegistry
from env_mock_agent.schemas import RuntimeName


class RuntimeRoutingDecision(BaseModel):
    selected: RuntimeName | None = None
    reason: str
    skipped: dict[str, str] = Field(default_factory=dict)
    runtime_version: str | None = None
    tools: list[str] = Field(default_factory=list)


class RuntimeRoutingRequest(BaseModel):
    artifact_type: str = ""
    role: str = ""
    required_tools: list[str] = Field(default_factory=list)
    preferred: list[RuntimeName] = Field(default_factory=list)
    previous_failures: dict[str, str] = Field(default_factory=dict)
    require_resume: bool = False


class RuntimeRouter:
    DEFAULT_ORDER = (
        RuntimeName.CLAUDE_AGENT_SDK,
        RuntimeName.CLAUDE_CODE_CLI,
        RuntimeName.PI_RPC,
    )

    def __init__(self, registry: RuntimeRegistry) -> None:
        self.registry = registry

    async def select(
        self,
        preferred: list[RuntimeName] | None = None,
        routing_request: RuntimeRoutingRequest | None = None,
    ) -> tuple[AgentRuntime | None, RuntimeRoutingDecision]:
        request = routing_request or RuntimeRoutingRequest()
        order = preferred or request.preferred or list(self.DEFAULT_ORDER)
        skipped: dict[str, str] = {}
        for name in order:
            if name.value in request.previous_failures:
                skipped[name.value] = "previous failure: " + request.previous_failures[name.value]
                continue
            try:
                runtime = self.registry.get(name)
            except KeyError:
                skipped[name.value] = "runtime not registered"
                continue
            try:
                capabilities = await runtime.probe()
            except Exception:
                skipped[name.value] = "runtime probe failed"
                continue
            if not capabilities.available:
                skipped[name.value] = capabilities.reason or "unavailable"
                continue
            if request.require_resume and not capabilities.supports_resume:
                skipped[name.value] = "runtime does not support cross-process session resume"
                continue
            available_tools = {tool.lower() for tool in capabilities.tools}
            missing_tools = sorted(
                tool for tool in request.required_tools if tool.lower() not in available_tools
            )
            if missing_tools:
                skipped[name.value] = f"missing required tools: {', '.join(missing_tools)}"
                continue
            return runtime, RuntimeRoutingDecision(
                selected=name,
                reason=f"{name.value} is the first available runtime in routing order",
                skipped=skipped,
                runtime_version=capabilities.version,
                tools=capabilities.tools,
            )
        return None, RuntimeRoutingDecision(
            reason="No open-ended runtime is available; return BLOCKED_CAPABILITY.",
            skipped=skipped,
        )
