from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.agent_system.private_store import FactoryPrivateObjectStore
from eval_factory.ai_gateway.invocation import EmbeddedAIGateway, ProviderInvocationResult
from eval_factory.ai_gateway.model_catalog import ModelCatalog
from eval_factory.ai_gateway.prompt_registry import PromptRegistry
from eval_factory.ai_gateway.receipts import GatewayRecordStore
from eval_factory.ai_gateway.routing import ModelRouter
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationRequestV2,
    GatewayInvocationStatusV2,
    GatewayUsageV2,
    ModelCapabilityProfileV2,
    ModelHealthSnapshotV2,
    ModelPriceScheduleV2,
    ModelQualityBaselineV2,
    ModelRoutingPolicyV2,
    PromptTemplateV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.harness import (
    GatewayJournalStateV1,
    GatewayRequirementAgentConfig,
    GatewayRequirementAgentLoop,
    HarnessSessionService,
    HarnessSessionStore,
    HarnessTurnOutcomeV1,
    ProviderEvidenceClassV1,
    RequirementInterpretationProposalV1,
)

HASH = "a" * 64
NOW = datetime(2026, 8, 17, tzinfo=UTC)


def _ref(object_type: str, suffix: str = "example", *, version: str = "v2") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/{version}",
        object_version=version,
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="harness-agent-loop-test",
        governing_versions=(VersionBinding(component="evaluation-agent-harness", version="v1", sha256=HASH),),
    )


class _ProposalProvider:
    def __init__(
        self,
        private_store: FactoryPrivateObjectStore,
        proposal: RequirementInterpretationProposalV1,
    ) -> None:
        self.private_store = private_store
        self.proposal = proposal
        self.calls = 0

    async def invoke(
        self,
        request: GatewayInvocationRequestV2,
        *,
        model_profile: ModelCapabilityProfileV2,
    ) -> ProviderInvocationResult:
        del request, model_profile
        self.calls += 1
        output_ref = self.private_store.put_model(
            object_type="requirement-proposal",
            value=self.proposal,
        )
        return ProviderInvocationResult(
            status=GatewayInvocationStatusV2.SUCCEEDED,
            response_body_ref=self.private_store.put_text(
                object_type="model-response-content",
                text="private fixture response",
            ),
            output_ref=output_ref,
            usage=GatewayUsageV2(
                input_tokens=100,
                output_tokens=50,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                charged_tokens=150,
                reported_cost_micro_usd=500,
            ),
            failure_code=None,
        )


def _components(
    root: Path,
    proposal: RequirementInterpretationProposalV1,
) -> tuple[
    HarnessSessionService,
    HarnessSessionStore,
    GatewayRequirementAgentLoop,
    _ProposalProvider,
]:
    private_store = FactoryPrivateObjectStore(root / "private")
    profile = ModelCapabilityProfileV2.create(
        model_profile_id="model-profile://harness-requirement",
        provider_id="provider://mechanism-fixture",
        model_id="model://harness-requirement",
        model_version="2026-08",
        capabilities=("requirement-interpretation", "structured-output"),
        supported_data_classifications=("INTERNAL",),
        supported_residencies=("LOCAL",),
        context_limit_tokens=100_000,
        output_limit_tokens=8_000,
        availability="AVAILABLE",
        audit=_audit(),
    )
    prompt = PromptTemplateV2.create(
        prompt_template_id="prompt-template://harness-requirement/v1",
        agent_role="requirement-agent",
        task_kind="harness-requirement-interpretation",
        system_template_ref=_ref("prompt-template-content", "system"),
        instruction_template_ref=_ref("prompt-template-content", "instruction"),
        required_input_object_types=("private-requirement-intake",),
        output_schema_ref=_ref("json-schema", "requirement-proposal"),
        allowed_tool_ids=(),
        required_model_capabilities=(
            "requirement-interpretation",
            "structured-output",
        ),
        injection_policy_ref=_ref("prompt-injection-policy"),
        template_version=1,
        audit=_audit(),
    )
    agent_definition_ref = _ref("agent-definition", "harness-requirement")
    policy = ModelRoutingPolicyV2.create(
        routing_policy_id="model-routing-policy://harness-requirement",
        allowed_provider_ids=("provider://mechanism-fixture",),
        allowed_agent_definition_refs=(agent_definition_ref,),
        allowed_model_profile_refs=(profile.to_ref(),),
        quality_weight=100,
        health_weight=10,
        latency_weight=1,
        cost_weight=1,
        audit=_audit(),
    )
    catalog = ModelCatalog(
        profiles=(profile,),
        quality_baselines=(
            ModelQualityBaselineV2.create(
                baseline_id="quality-baseline://harness-requirement",
                model_profile_ref=profile.to_ref(),
                task_kind="harness-requirement-interpretation",
                benchmark_ref=_ref("model-quality-benchmark"),
                quality_basis_points=9000,
                minimum_sample_count=10,
                audit=_audit(),
            ),
        ),
        health_snapshots=(
            ModelHealthSnapshotV2.create(
                health_snapshot_id="health://harness-requirement",
                model_profile_ref=profile.to_ref(),
                observed_at=NOW,
                availability="HEALTHY",
                success_basis_points=9900,
                latency_milliseconds=100,
                audit=_audit(),
            ),
        ),
        price_schedules=(
            ModelPriceScheduleV2.create(
                price_schedule_id="price://harness-requirement",
                model_profile_ref=profile.to_ref(),
                input_micro_usd_per_million_tokens=1_000,
                output_micro_usd_per_million_tokens=2_000,
                effective_from=NOW,
                audit=_audit(),
            ),
        ),
    )
    records = GatewayRecordStore(root / "gateway.sqlite3")
    provider = _ProposalProvider(private_store, proposal)
    gateway = EmbeddedAIGateway(
        router=ModelRouter(catalog, policy),
        catalog=catalog,
        prompts=PromptRegistry((prompt,)),
        records=records,
        provider=provider,
        clock=lambda: NOW,
    )
    session_store = HarnessSessionStore(
        root / "session.sqlite3",
        clock=lambda: NOW,
    )
    agent = GatewayRequirementAgentLoop(
        gateway=gateway,
        lookup=records,
        private_store=private_store,
        session_store=session_store,
        config=GatewayRequirementAgentConfig(
            prompt=prompt,
            agent_definition_ref=agent_definition_ref,
            allowed_model_profile_refs=(profile.to_ref(),),
            budget_reservation_ref=_ref("work-model-reservation"),
            data_classification="INTERNAL",
            residency="LOCAL",
            input_token_budget=4_000,
            output_token_budget=2_000,
            max_cost_micro_usd=1_000_000,
            minimum_quality_basis_points=8000,
            evidence_class=ProviderEvidenceClassV1.MECHANISM_FIXTURE,
            assistant_chunk_characters=64,
        ),
    )
    return (
        HarnessSessionService(store=session_store, requirement_agent=agent),
        session_store,
        agent,
        provider,
    )


@pytest.mark.asyncio
async def test_gateway_agent_ready_turn_streams_committed_events_and_replays(
    tmp_path: Path,
) -> None:
    proposal = RequirementInterpretationProposalV1(
        outcome="READY",
        assistant_message="需求完整, 已形成通用 Agent 评测数据生产解释。",
        goals=("构建通用 Agent 评测数据",),
        source_expectations=("用户提供执行轨迹",),
        target_capabilities=("agent-evaluation",),
        quality_intent="可审计且无答案泄漏",
        delivery_intent="候选数据包",
        max_model_requests=10,
        max_model_tokens=100_000,
        max_cost_micro_usd=1_000_000,
    )
    service, store, _, provider = _components(tmp_path, proposal)
    created = service.create_session(
        session_id="session-agent-ready",
        incarnation_id="session-agent-ready-incarnation",
        composition_ref=_ref("harness-composition", version="v1"),
        created_by="agent-user",
        idempotency_key="create-agent-ready",
        audit=_audit(),
    )
    events = [
        event
        async for event in service.stream_turn(
            session_id=created.session.session_id,
            expected_session_version=created.session.session_version,
            principal_ref=_ref("principal", version="v1"),
            content="请根据执行轨迹生产通用 Agent 评测数据。",
            artifact_envelope_refs=(),
            idempotency_key="turn-agent-ready",
            audit=_audit(),
        )
    ]
    assert provider.calls == 1
    assert events[0].payload.family == "MESSAGE"
    assert events[-1].payload.family == "REQUIREMENT"

    current = service.get_session("session-agent-ready")
    assert current.current_requirement_policy_ref is not None
    assert current.latest_gateway is not None
    assert current.latest_gateway.evidence_class is (ProviderEvidenceClassV1.MECHANISM_FIXTURE)
    assert current.latest_gateway.usage is not None
    assert current.latest_gateway.usage.reported_cost_micro_usd == 500
    assert "private fixture response" not in current.model_dump_json()

    replay = await service.post_message(
        session_id="session-agent-ready",
        expected_session_version=1,
        principal_ref=_ref("principal", version="v1"),
        content="请根据执行轨迹生产通用 Agent 评测数据。",
        artifact_envelope_refs=(),
        idempotency_key="turn-agent-ready",
        audit=_audit(),
    )
    assert replay.outcome is HarnessTurnOutcomeV1.READY
    assert provider.calls == 1

    reopened = HarnessSessionStore(tmp_path / "session.sqlite3", clock=lambda: NOW)
    assert reopened.get_projection("session-agent-ready") == current
    assert store.list_events("session-agent-ready", after_sequence=0, limit=100).events


def test_prepared_without_gateway_result_reconciles_to_unknown_without_call(
    tmp_path: Path,
) -> None:
    proposal = RequirementInterpretationProposalV1(
        outcome="CLARIFICATION_REQUIRED",
        assistant_message="请补充数据来源。",
        missing_field_codes=("SOURCE_EXPECTATION_MISSING",),
        clarification_questions=("数据来源是什么?",),
    )
    service, store, agent, provider = _components(tmp_path, proposal)
    created = service.create_session(
        session_id="session-agent-unknown",
        incarnation_id="session-agent-unknown-incarnation",
        composition_ref=_ref("harness-composition", version="v1"),
        created_by="agent-user",
        idempotency_key="create-agent-unknown",
        audit=_audit(),
    )
    start = store.start_turn(
        session_id=created.session.session_id,
        expected_session_version=created.session.session_version,
        principal_ref=_ref("principal", version="v1"),
        content="做一个评测集。",
        artifact_envelope_refs=(),
        idempotency_key="turn-agent-unknown",
        audit=_audit(),
    )
    agent.prepare(start, audit=_audit())
    result = service.reconcile(start.command.to_ref(), audit=_audit())
    assert result.outcome is HarnessTurnOutcomeV1.UNKNOWN_OUTCOME
    assert provider.calls == 0
    current = service.get_session("session-agent-unknown")
    assert current.latest_gateway is not None
    assert current.latest_gateway.state is GatewayJournalStateV1.UNKNOWN_OUTCOME


@pytest.mark.asyncio
async def test_gateway_agent_persists_clarification_without_policy(
    tmp_path: Path,
) -> None:
    proposal = RequirementInterpretationProposalV1(
        outcome="CLARIFICATION_REQUIRED",
        assistant_message="请补充数据来源。",
        missing_field_codes=("SOURCE_EXPECTATION_MISSING",),
        clarification_questions=("数据来源是什么?",),
    )
    service, _, _, provider = _components(tmp_path, proposal)
    created = service.create_session(
        session_id="session-agent-clarification",
        incarnation_id="session-agent-clarification-incarnation",
        composition_ref=_ref("harness-composition", version="v1"),
        created_by="agent-user",
        idempotency_key="create-agent-clarification",
        audit=_audit(),
    )
    result = await service.post_message(
        session_id=created.session.session_id,
        expected_session_version=created.session.session_version,
        principal_ref=_ref("principal", version="v1"),
        content="做一个评测集。",
        artifact_envelope_refs=(),
        idempotency_key="turn-agent-clarification",
        audit=_audit(),
    )
    assert result.outcome is HarnessTurnOutcomeV1.CLARIFICATION_REQUIRED
    assert result.requirement_policy_ref is None
    assert provider.calls == 1
    current = service.get_session(created.session.session_id)
    assert current.pending_clarification_questions == ("数据来源是什么?",)


def test_requirement_agent_config_rejects_invalid_runtime_limits(
    tmp_path: Path,
) -> None:
    proposal = RequirementInterpretationProposalV1(
        outcome="CLARIFICATION_REQUIRED",
        assistant_message="请补充数据来源。",
        missing_field_codes=("SOURCE_EXPECTATION_MISSING",),
        clarification_questions=("数据来源是什么?",),
    )
    _, _, agent, _ = _components(tmp_path, proposal)
    with pytest.raises(ValueError, match="chunk size"):
        replace(agent.config, assistant_chunk_characters=1)
    with pytest.raises(ValueError, match="model profile"):
        replace(agent.config, allowed_model_profile_refs=())
