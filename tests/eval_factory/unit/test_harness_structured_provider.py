from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.agent_system.private_store import FactoryPrivateObjectStore
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationRequestV2,
    ModelCapabilityProfileV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.harness.runtime_models import (
    RequirementIntakeV1,
    RequirementInterpretationProposalV1,
)
from eval_factory.harness.structured_provider import (
    RequirementStructuredGatewayProvider,
    RequirementStructuredProviderConfig,
    RequirementStructuredProviderError,
)

HASH = "a" * 64
NOW = datetime(2026, 8, 17, tzinfo=UTC)


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
    digest: str = HASH,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="structured-provider-test",
        governing_versions=(
            VersionBinding(
                component="evaluation-agent-harness",
                version="v1",
                sha256=HASH,
            ),
        ),
    )


class _Client:
    def __init__(
        self,
        proposal: RequirementInterpretationProposalV1,
        *,
        usage: dict[str, object] | None = None,
    ) -> None:
        self.proposal = proposal
        self.usage = usage or {
            "input_tokens": 30,
            "output_tokens": 20,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 5,
        }
        self.calls: list[dict[str, object]] = []

    async def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, object],
        max_output_tokens: int = 64_000,
    ) -> tuple[dict[str, object], dict[str, object]]:
        self.calls.append(
            {
                "max_output_tokens": max_output_tokens,
                "prompt": prompt,
                "schema": schema,
                "system": system,
            }
        )
        return self.proposal.model_dump(mode="json"), self.usage


def _proposal() -> RequirementInterpretationProposalV1:
    return RequirementInterpretationProposalV1(
        outcome="READY",
        assistant_message="需求完整, 已形成评测数据生产解释。",
        goals=("构建通用 Agent 评测数据",),
        source_expectations=("使用已准入的脱敏 Agent 轨迹",),
        target_capabilities=("generic-agent-trace",),
        quality_intent="可审计且无答案泄漏",
        delivery_intent="JSONL 候选数据包",
        max_model_requests=12,
        max_model_tokens=120_000,
        max_cost_micro_usd=2_000_000,
    )


def _profile(
    *,
    provider_id: str = "provider://anthropic",
    model_id: str = "claude-opus-4-7",
) -> ModelCapabilityProfileV2:
    return ModelCapabilityProfileV2.create(
        model_profile_id="model-profile://harness-real-semantic",
        provider_id=provider_id,
        model_id=model_id,
        model_version="claude-opus-4-7",
        capabilities=("requirement-interpretation", "structured-output"),
        supported_data_classifications=("INTERNAL",),
        supported_residencies=("REMOTE_APPROVED",),
        context_limit_tokens=1_000_000,
        output_limit_tokens=16_000,
        availability="AVAILABLE",
        audit=_audit(),
    )


def _provider(
    root: Path,
    client: _Client,
    *,
    schema_payload: bytes | None = None,
) -> tuple[
    RequirementStructuredGatewayProvider,
    FactoryPrivateObjectStore,
    ObjectRef,
]:
    private_store = FactoryPrivateObjectStore(root)
    system_ref = private_store.put_text(
        object_type="prompt-template-content",
        text="Interpret evaluation-data requirements.",
    )
    instruction_ref = private_store.put_text(
        object_type="prompt-template-content",
        text="Return one strict requirement proposal.",
    )
    schema_ref = private_store.put_bytes(
        object_type="json-schema",
        payload=(
            schema_payload
            if schema_payload is not None
            else (
                json.dumps(
                    RequirementInterpretationProposalV1.model_json_schema(
                        mode="validation",
                    ),
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode()
        ),
    )
    provider = RequirementStructuredGatewayProvider(
        client=client,
        private_store=private_store,
        config=RequirementStructuredProviderConfig(
            provider_id="provider://anthropic",
            model_id="claude-opus-4-7",
            system_template_ref=system_ref,
            instruction_template_ref=instruction_ref,
            output_schema_ref=schema_ref,
            max_output_tokens=2_000,
        ),
    )
    return provider, private_store, schema_ref


def _request(
    private_store: FactoryPrivateObjectStore,
    schema_ref: ObjectRef,
) -> GatewayInvocationRequestV2:
    intake = RequirementIntakeV1(
        session_ref=_ref("harness-session", "real", version="v1"),
        command_ref=_ref("session-command", "real", version="v1"),
        user_message_ref=_ref("harness-message", "real", version="v1"),
        user_text="构建一套通用 Agent 评测数据。",
    )
    rendering_ref = private_store.put_model(
        object_type="prompt-rendering",
        value=intake,
    )
    return GatewayInvocationRequestV2.create(
        invocation_request_id="gateway-invocation-request://structured-provider",
        agent_task_ref=_ref("harness-turn", "real", version="v1"),
        route_decision_ref=_ref("model-route-decision", "real"),
        prompt_template_ref=_ref("prompt-template", "real"),
        prompt_rendering_ref=rendering_ref,
        output_schema_ref=schema_ref,
        rag_result_refs=(),
        idempotency_key="structured-provider-request",
        audit=_audit(),
    )


@pytest.mark.asyncio
async def test_structured_provider_persists_private_output_and_usage(
    tmp_path: Path,
) -> None:
    client = _Client(_proposal())
    provider, private_store, schema_ref = _provider(tmp_path, client)

    result = await provider.invoke(
        _request(private_store, schema_ref),
        model_profile=_profile(),
    )

    assert provider.invocation_count == 1
    assert result.output_ref is not None
    assert result.response_body_ref is not None
    assert result.usage.charged_tokens == 65
    assert result.usage.reported_cost_micro_usd is None
    assert (
        private_store.get_model(
            result.output_ref,
            RequirementInterpretationProposalV1,
        )
        == _proposal()
    )
    assert client.calls[0]["max_output_tokens"] == 2_000
    assert "构建一套通用 Agent 评测数据" in str(client.calls[0]["prompt"])


@pytest.mark.asyncio
async def test_structured_provider_rejects_profile_or_schema_drift_before_call(
    tmp_path: Path,
) -> None:
    client = _Client(_proposal())
    provider, private_store, schema_ref = _provider(tmp_path, client)
    request = _request(private_store, schema_ref)

    with pytest.raises(RequirementStructuredProviderError, match="model profile"):
        await provider.invoke(
            request,
            model_profile=_profile(model_id="another-model"),
        )
    changed_schema_ref = private_store.put_bytes(
        object_type="json-schema",
        payload=b'{"type":"object"}',
    )
    with pytest.raises(RequirementStructuredProviderError, match="output schema"):
        await provider.invoke(
            request.model_copy(update={"output_schema_ref": changed_schema_ref}),
            model_profile=_profile(),
        )
    assert client.calls == []
    assert provider.invocation_count == 0


@pytest.mark.asyncio
async def test_structured_provider_rejects_invalid_private_schema(
    tmp_path: Path,
) -> None:
    client = _Client(_proposal())
    provider, private_store, schema_ref = _provider(
        tmp_path,
        client,
        schema_payload=b"[]",
    )

    with pytest.raises(RequirementStructuredProviderError, match="must be an object"):
        await provider.invoke(
            _request(private_store, schema_ref),
            model_profile=_profile(),
        )
    assert provider.invocation_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    (
        {"input_tokens": True},
        {"input_tokens": -1},
        {"reported_cost_micro_usd": False},
        {"reported_cost_micro_usd": -1},
    ),
)
async def test_structured_provider_rejects_invalid_usage(
    tmp_path: Path,
    usage: dict[str, object],
) -> None:
    client = _Client(_proposal(), usage=usage)
    provider, private_store, schema_ref = _provider(tmp_path, client)

    with pytest.raises(RequirementStructuredProviderError, match="provider"):
        await provider.invoke(
            _request(private_store, schema_ref),
            model_profile=_profile(),
        )


def test_structured_provider_config_rejects_invalid_refs_and_budget(
    tmp_path: Path,
) -> None:
    store = FactoryPrivateObjectStore(tmp_path)
    text_ref = store.put_text(
        object_type="prompt-template-content",
        text="template",
    )
    schema_ref = store.put_bytes(
        object_type="json-schema",
        payload=b'{"type":"object"}',
    )
    with pytest.raises(ValueError, match="system_template_ref"):
        RequirementStructuredProviderConfig(
            provider_id="provider://anthropic",
            model_id="claude-opus-4-7",
            system_template_ref=schema_ref,
            instruction_template_ref=text_ref,
            output_schema_ref=schema_ref,
        )
    with pytest.raises(ValueError, match="output budget"):
        RequirementStructuredProviderConfig(
            provider_id="provider://anthropic",
            model_id="claude-opus-4-7",
            system_template_ref=text_ref,
            instruction_template_ref=text_ref,
            output_schema_ref=schema_ref,
            max_output_tokens=128,
        )
