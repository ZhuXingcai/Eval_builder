from __future__ import annotations

import pytest

from env_mock_agent.runtimes import (
    FakeRuntime,
    RuntimeRegistry,
    RuntimeRouter,
    RuntimeRoutingRequest,
)
from env_mock_agent.schemas import RuntimeName


class _ProbeFailureRuntime(FakeRuntime):
    async def probe(self):
        raise RuntimeError("private probe failure detail")


@pytest.mark.asyncio
async def test_router_selects_runtime_that_satisfies_required_tools() -> None:
    registry = RuntimeRegistry()
    registry.register(RuntimeName.FAKE, FakeRuntime())
    runtime, decision = await RuntimeRouter(registry).select(
        routing_request=RuntimeRoutingRequest(
            preferred=[RuntimeName.FAKE],
            required_tools=["write"],
            require_resume=True,
        )
    )
    assert runtime is not None
    assert decision.selected == RuntimeName.FAKE
    assert decision.runtime_version == "1"


@pytest.mark.asyncio
async def test_router_records_capability_and_failure_skips() -> None:
    registry = RuntimeRegistry()
    registry.register(RuntimeName.FAKE, FakeRuntime())
    runtime, decision = await RuntimeRouter(registry).select(
        routing_request=RuntimeRoutingRequest(
            preferred=[RuntimeName.FAKE],
            required_tools=["web_search"],
        )
    )
    assert runtime is None
    assert decision.selected is None
    assert decision.skipped["fake"] == "missing required tools: web_search"

    runtime, decision = await RuntimeRouter(registry).select(
        routing_request=RuntimeRoutingRequest(
            preferred=[RuntimeName.FAKE],
            previous_failures={"fake": "authentication failed"},
        )
    )
    assert runtime is None
    assert decision.skipped["fake"] == "previous failure: authentication failed"


@pytest.mark.asyncio
async def test_router_skips_unregistered_runtime_and_selects_later_candidate() -> None:
    registry = RuntimeRegistry()
    registry.register(RuntimeName.FAKE, FakeRuntime())

    runtime, decision = await RuntimeRouter(registry).select(
        preferred=[RuntimeName.CLAUDE_AGENT_SDK, RuntimeName.FAKE]
    )

    assert runtime is not None
    assert decision.selected == RuntimeName.FAKE
    assert decision.skipped == {"claude_agent_sdk": "runtime not registered"}


@pytest.mark.asyncio
async def test_router_skips_probe_failure_without_disclosing_exception() -> None:
    registry = RuntimeRegistry()
    registry.register(RuntimeName.CLAUDE_AGENT_SDK, _ProbeFailureRuntime())
    registry.register(RuntimeName.FAKE, FakeRuntime())

    runtime, decision = await RuntimeRouter(registry).select(
        preferred=[RuntimeName.CLAUDE_AGENT_SDK, RuntimeName.FAKE]
    )

    assert runtime is not None
    assert decision.selected == RuntimeName.FAKE
    assert decision.skipped == {"claude_agent_sdk": "runtime probe failed"}
    assert "private probe failure detail" not in str(decision.model_dump(mode="json"))
