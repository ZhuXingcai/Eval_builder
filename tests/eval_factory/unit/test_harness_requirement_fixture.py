from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError
from tests.eval_factory.integration.test_harness_agent_loop import (
    _audit,
    _ref,
)

from eval_factory.ai_gateway.invocation import (
    EmbeddedAIGateway,
)
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayUsageV2,
    ModelCapabilityProfileV2,
    ModelHealthSnapshotV2,
    ModelPriceScheduleV2,
    ModelQualityBaselineV2,
    ModelRoutingPolicyV2,
    PromptTemplateV2,
)
from eval_factory.harness import (
    FixtureRequirementAgentConfigV1,
    FixtureRequirementGatewayProvider,
    HarnessTurnOutcomeV1,
    RequirementInterpretationProposalV1,
    build_fixture_requirement_session_service,
)


def fixture_requirement_config() -> FixtureRequirementAgentConfigV1:
    profile = ModelCapabilityProfileV2.create(
        model_profile_id="model-profile://harness-requirement",
        provider_id="provider://mechanism-fixture",
        model_id="model://harness-requirement",
        model_version="2026-08",
        capabilities=(
            "requirement-interpretation",
            "structured-output",
        ),
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
        system_template_ref=_ref(
            "prompt-template-content",
            "system",
        ),
        instruction_template_ref=_ref(
            "prompt-template-content",
            "instruction",
        ),
        required_input_object_types=("private-requirement-intake",),
        output_schema_ref=_ref(
            "json-schema",
            "requirement-proposal",
        ),
        allowed_tool_ids=(),
        required_model_capabilities=(
            "requirement-interpretation",
            "structured-output",
        ),
        injection_policy_ref=_ref("prompt-injection-policy"),
        template_version=1,
        audit=_audit(),
    )
    agent_ref = _ref(
        "agent-definition",
        "harness-requirement",
    )
    return FixtureRequirementAgentConfigV1(
        model_profile=profile,
        quality_baseline=ModelQualityBaselineV2.create(
            baseline_id="quality-baseline://harness-requirement",
            model_profile_ref=profile.to_ref(),
            task_kind=prompt.task_kind,
            benchmark_ref=_ref("model-quality-benchmark"),
            quality_basis_points=9_000,
            minimum_sample_count=10,
            audit=_audit(),
        ),
        health_snapshot=ModelHealthSnapshotV2.create(
            health_snapshot_id="health://harness-requirement",
            model_profile_ref=profile.to_ref(),
            observed_at=_audit().created_at,
            availability="HEALTHY",
            success_basis_points=9_900,
            latency_milliseconds=100,
            audit=_audit(),
        ),
        price_schedule=ModelPriceScheduleV2.create(
            price_schedule_id="price://harness-requirement",
            model_profile_ref=profile.to_ref(),
            input_micro_usd_per_million_tokens=1_000,
            output_micro_usd_per_million_tokens=2_000,
            effective_from=_audit().created_at,
            audit=_audit(),
        ),
        routing_policy=ModelRoutingPolicyV2.create(
            routing_policy_id=("model-routing-policy://harness-requirement"),
            allowed_provider_ids=(profile.provider_id,),
            allowed_agent_definition_refs=(agent_ref,),
            allowed_model_profile_refs=(profile.to_ref(),),
            quality_weight=100,
            health_weight=10,
            latency_weight=1,
            cost_weight=1,
            audit=_audit(),
        ),
        prompt=prompt,
        agent_definition_ref=agent_ref,
        budget_reservation_ref=_ref("work-model-reservation"),
        proposal=RequirementInterpretationProposalV1(
            outcome="READY",
            assistant_message="Requirement is ready.",
            goals=("Build generic Agent evaluation data.",),
            source_expectations=("Use the admitted trace manifest.",),
            target_capabilities=("generic-agent-trace",),
            quality_intent="Auditable candidate data.",
            delivery_intent="Candidate JSON and JSONL.",
            max_model_requests=100,
            max_model_tokens=1_000_000,
            max_cost_micro_usd=10_000_000,
        ),
        usage=GatewayUsageV2(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            charged_tokens=150,
            reported_cost_micro_usd=500,
        ),
        data_classification="INTERNAL",
        residency="LOCAL",
        input_token_budget=4_000,
        output_token_budget=2_000,
        max_cost_micro_usd=1_000_000,
        minimum_quality_basis_points=8_000,
        assistant_chunk_characters=64,
    )


@pytest.mark.asyncio
async def test_fixture_requirement_service_replays_without_provider_call(
    tmp_path: Path,
) -> None:
    config = fixture_requirement_config()
    values = {
        "harness_store_path": tmp_path / "harness.sqlite3",
        "private_store_path": tmp_path / "private",
        "gateway_store_path": tmp_path / "gateway.sqlite3",
        "config": config,
    }
    service = build_fixture_requirement_session_service(**values)
    created = service.create_session(
        session_id="session-fixture-requirement",
        incarnation_id="session-fixture-requirement-incarnation",
        composition_ref=_ref(
            "harness-composition",
            version="v1",
        ),
        created_by="user://fixture",
        idempotency_key="create-session-fixture-requirement",
        audit=_audit(),
    )
    command = {
        "session_id": created.session.session_id,
        "expected_session_version": created.session.session_version,
        "principal_ref": _ref("principal"),
        "content": "Build the evaluation dataset.",
        "artifact_envelope_refs": (),
        "idempotency_key": "message-fixture-requirement",
        "audit": _audit(),
    }
    first = await service.post_message(**command)
    provider = cast(
        FixtureRequirementGatewayProvider,
        cast(EmbeddedAIGateway, service.requirement_agent.gateway)._provider,
    )
    assert first.outcome is HarnessTurnOutcomeV1.READY
    assert provider.invocation_count == 1

    restarted = build_fixture_requirement_session_service(**values)
    replay = await restarted.post_message(**command)
    restarted_provider = cast(
        FixtureRequirementGatewayProvider,
        cast(
            EmbeddedAIGateway,
            restarted.requirement_agent.gateway,
        )._provider,
    )
    assert replay == first
    assert restarted_provider.invocation_count == 0


def test_fixture_requirement_config_rejects_authority_drift() -> None:
    config = fixture_requirement_config()
    with pytest.raises(ValidationError, match="inconsistent"):
        FixtureRequirementAgentConfigV1.model_validate(
            {
                **config.model_dump(mode="python"),
                "agent_definition_ref": _ref(
                    "agent-definition",
                    "changed",
                ),
            }
        )
