from __future__ import annotations

from datetime import UTC, datetime

import pytest

from env_mock_agent.facade import (
    AttachmentRouteCandidateKind,
    AttachmentRouteOutcome,
    AttachmentRouteRequestV2,
    AttachmentRouteSkipCode,
    FacadeObjectRef,
    RegistryAttachmentRoutingFacade,
    attachment_route_request_carried_sha256,
)
from env_mock_agent.providers import (
    ProviderCapability,
    ProviderRegistry,
    TextProvider,
)
from env_mock_agent.runtimes import FakeRuntime, RuntimeRegistry
from env_mock_agent.schemas import RuntimeCapabilities, RuntimeName

HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=HASH,
    )


def _request(**overrides: object) -> AttachmentRouteRequestV2:
    values: dict[str, object] = {
        "route_request_id": "attachment-route-request://pending",
        "artifact_id": "artifact://routing/input",
        "build_contract_ref": _ref(
            "artifact-build-contract",
            "input",
            version="v2",
        ),
        "routing_policy_ref": _ref(
            "artifact-routing-policy",
            "current",
            version="v2",
        ),
        "asset_type": "txt",
        "media_type": "text/plain",
        "mode": "PROMPT_ONLY",
        "criticality": "REQUIRED",
        "provider_payload_ref": _ref(
            "attachment-provider-payload",
            "input",
            version="v2",
        ),
        "approved_provider_ids": ("text",),
        "required_provider_capability_ids": ("attachment-provider/generate/txt/v1",),
        "runtime_order": (
            "claude_agent_sdk",
            "claude_code_cli",
            "pi_rpc",
        ),
        "required_runtime_tools": ("write",),
        "require_runtime_resume": True,
        "idempotency_key": "attachment-route-idempotency://input",
        "route_request_sha256": HASH,
    }
    values.update(overrides)
    request = AttachmentRouteRequestV2(**values)
    digest = attachment_route_request_carried_sha256(request)
    return request.model_copy(
        update={
            "route_request_id": (f"attachment-route-request://sha256/{digest}"),
            "route_request_sha256": digest,
        }
    )


class _CountingRuntime(FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.probe_count = 0

    async def probe(self):
        self.probe_count += 1
        return await super().probe()


class _StaticRuntime(FakeRuntime):
    def __init__(
        self,
        *,
        tools: tuple[str, ...],
        supports_resume: bool,
    ) -> None:
        super().__init__()
        self._tools = tools
        self._supports_resume = supports_resume

    async def probe(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            name=RuntimeName.FAKE,
            available=True,
            version="static-1",
            tools=list(self._tools),
            supports_resume=self._supports_resume,
        )


class _ProbeFailureRuntime(FakeRuntime):
    async def probe(self) -> RuntimeCapabilities:
        raise RuntimeError("private runtime probe detail")


class _UnavailableTextProvider(TextProvider):
    def probe(self) -> ProviderCapability:
        return ProviderCapability(
            name=self.name,
            asset_types=list(self.asset_types),
            available=False,
            reason="private provider unavailable detail",
        )


class _ProbeFailureTextProvider(TextProvider):
    def probe(self) -> ProviderCapability:
        raise RuntimeError("private provider probe detail")


@pytest.mark.asyncio
async def test_deterministic_provider_wins_without_runtime_probe() -> None:
    providers = ProviderRegistry()
    providers.register(TextProvider())
    runtime = _CountingRuntime()
    runtimes = RuntimeRegistry()
    runtimes.register(RuntimeName.CLAUDE_AGENT_SDK, runtime)
    facade = RegistryAttachmentRoutingFacade(
        providers,
        runtimes,
        clock=lambda: datetime(2026, 7, 28, tzinfo=UTC),
    )

    decision = await facade.route(_request())

    assert decision.outcome is AttachmentRouteOutcome.SELECTED_PROVIDER
    assert decision.selected_kind is AttachmentRouteCandidateKind.PROVIDER
    assert decision.selected_id == "text"
    assert decision.satisfied_capability_ids == ("attachment-provider/generate/txt/v1",)
    assert decision.skipped == ()
    assert runtime.probe_count == 0


@pytest.mark.asyncio
async def test_missing_provider_payload_records_skip_then_selects_sdk() -> None:
    providers = ProviderRegistry()
    providers.register(TextProvider())
    runtimes = RuntimeRegistry()
    runtimes.register(RuntimeName.CLAUDE_AGENT_SDK, FakeRuntime())
    facade = RegistryAttachmentRoutingFacade(
        providers,
        runtimes,
        clock=lambda: datetime(2026, 7, 28, tzinfo=UTC),
    )

    decision = await facade.route(_request(provider_payload_ref=None))

    assert decision.outcome is AttachmentRouteOutcome.SELECTED_RUNTIME
    assert decision.selected_id == "claude_agent_sdk"
    assert tuple(item.code for item in decision.skipped) == (
        AttachmentRouteSkipCode.PROVIDER_PAYLOAD_MISSING,
    )
    assert decision.satisfied_capability_ids == ("write",)


@pytest.mark.asyncio
async def test_all_missing_runtimes_return_content_free_block() -> None:
    providers = ProviderRegistry()
    providers.register(TextProvider())
    facade = RegistryAttachmentRoutingFacade(
        providers,
        RuntimeRegistry(),
        clock=lambda: datetime(2026, 7, 28, tzinfo=UTC),
    )

    decision = await facade.route(_request(provider_payload_ref=None))

    assert decision.outcome is AttachmentRouteOutcome.BLOCKED_CAPABILITY
    assert decision.selected_id is None
    assert tuple(item.code for item in decision.skipped) == (
        AttachmentRouteSkipCode.PROVIDER_PAYLOAD_MISSING,
        AttachmentRouteSkipCode.RUNTIME_NOT_REGISTERED,
        AttachmentRouteSkipCode.RUNTIME_NOT_REGISTERED,
        AttachmentRouteSkipCode.RUNTIME_NOT_REGISTERED,
    )
    serialized = str(decision.model_dump(mode="json"))
    for forbidden in (
        "private probe failure detail",
        "environment",
        "credential",
        "command",
        "workspace",
        "transcript",
        "provider_payload",
    ):
        assert forbidden not in serialized


@pytest.mark.asyncio
async def test_provider_capability_gap_falls_back_with_exact_missing_id() -> None:
    providers = ProviderRegistry()
    providers.register(TextProvider())
    runtimes = RuntimeRegistry()
    runtimes.register(RuntimeName.CLAUDE_AGENT_SDK, FakeRuntime())
    facade = RegistryAttachmentRoutingFacade(
        providers,
        runtimes,
        clock=lambda: datetime(2026, 7, 28, tzinfo=UTC),
    )

    decision = await facade.route(
        _request(required_provider_capability_ids=("attachment-provider/generate/xlsx/v1",))
    )

    assert decision.outcome is AttachmentRouteOutcome.SELECTED_RUNTIME
    assert decision.selected_id == "claude_agent_sdk"
    assert decision.skipped[0].code is (AttachmentRouteSkipCode.PROVIDER_CAPABILITY_MISSING)
    assert decision.skipped[0].missing_capability_ids == ("attachment-provider/generate/xlsx/v1",)


@pytest.mark.asyncio
async def test_runtime_tool_resume_and_registration_gaps_are_typed() -> None:
    providers = ProviderRegistry()
    providers.register(TextProvider())
    runtimes = RuntimeRegistry()
    runtimes.register(
        RuntimeName.CLAUDE_AGENT_SDK,
        _StaticRuntime(
            tools=("write",),
            supports_resume=True,
        ),
    )
    runtimes.register(
        RuntimeName.CLAUDE_CODE_CLI,
        _StaticRuntime(
            tools=("web_search",),
            supports_resume=False,
        ),
    )
    facade = RegistryAttachmentRoutingFacade(
        providers,
        runtimes,
        clock=lambda: datetime(2026, 7, 28, tzinfo=UTC),
    )

    decision = await facade.route(
        _request(
            provider_payload_ref=None,
            required_runtime_tools=("web_search",),
        )
    )

    assert decision.outcome is AttachmentRouteOutcome.BLOCKED_CAPABILITY
    assert tuple(item.code for item in decision.skipped) == (
        AttachmentRouteSkipCode.PROVIDER_PAYLOAD_MISSING,
        AttachmentRouteSkipCode.RUNTIME_TOOL_MISSING,
        AttachmentRouteSkipCode.RUNTIME_RESUME_UNSUPPORTED,
        AttachmentRouteSkipCode.RUNTIME_NOT_REGISTERED,
    )
    assert decision.skipped[1].missing_capability_ids == ("web_search",)


@pytest.mark.asyncio
async def test_runtime_probe_failure_is_sanitized_before_later_selection() -> None:
    providers = ProviderRegistry()
    providers.register(TextProvider())
    runtimes = RuntimeRegistry()
    runtimes.register(
        RuntimeName.CLAUDE_AGENT_SDK,
        _ProbeFailureRuntime(),
    )
    runtimes.register(RuntimeName.CLAUDE_CODE_CLI, FakeRuntime())
    facade = RegistryAttachmentRoutingFacade(
        providers,
        runtimes,
        clock=lambda: datetime(2026, 7, 28, tzinfo=UTC),
    )

    decision = await facade.route(_request(provider_payload_ref=None))

    assert decision.outcome is AttachmentRouteOutcome.SELECTED_RUNTIME
    assert decision.selected_id == "claude_code_cli"
    assert tuple(item.code for item in decision.skipped) == (
        AttachmentRouteSkipCode.PROVIDER_PAYLOAD_MISSING,
        AttachmentRouteSkipCode.RUNTIME_PROBE_FAILED,
    )
    assert "private runtime probe detail" not in str(decision.model_dump(mode="json"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "expected_code"),
    [
        (
            "unsupported",
            AttachmentRouteSkipCode.PROVIDER_UNSUPPORTED_ASSET_TYPE,
        ),
        (
            "unapproved",
            AttachmentRouteSkipCode.PROVIDER_NOT_APPROVED,
        ),
        (
            "unavailable",
            AttachmentRouteSkipCode.PROVIDER_UNAVAILABLE,
        ),
        (
            "probe-failed",
            AttachmentRouteSkipCode.PROVIDER_PROBE_FAILED,
        ),
    ],
)
async def test_provider_gaps_are_typed_before_runtime_fallback(
    scenario: str,
    expected_code: AttachmentRouteSkipCode,
) -> None:
    providers = ProviderRegistry()
    overrides: dict[str, object] = {}
    if scenario == "unsupported":
        providers.register(TextProvider())
        overrides.update(
            asset_type="csv",
            required_provider_capability_ids=("attachment-provider/generate/csv/v1",),
        )
    elif scenario == "unapproved":
        providers.register(TextProvider())
        overrides["approved_provider_ids"] = ("other-provider",)
    elif scenario == "unavailable":
        providers.register(_UnavailableTextProvider())
    else:
        providers.register(_ProbeFailureTextProvider())
    runtimes = RuntimeRegistry()
    runtimes.register(RuntimeName.CLAUDE_AGENT_SDK, FakeRuntime())
    facade = RegistryAttachmentRoutingFacade(
        providers,
        runtimes,
        clock=lambda: datetime(2026, 7, 28, tzinfo=UTC),
    )

    decision = await facade.route(_request(**overrides))

    assert decision.outcome is AttachmentRouteOutcome.SELECTED_RUNTIME
    assert decision.selected_id == "claude_agent_sdk"
    assert decision.skipped[0].code is expected_code
    serialized = str(decision.model_dump(mode="json"))
    assert "private provider unavailable detail" not in serialized
    assert "private provider probe detail" not in serialized
