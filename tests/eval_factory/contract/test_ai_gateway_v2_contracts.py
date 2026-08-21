from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ValidationError

import eval_factory.contracts.ai_gateway_v2 as contracts
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationRequestV2,
    GatewayInvocationResultV2,
    GatewayInvocationStatusV2,
    GatewayReceiptV2,
    GatewayUsageV2,
    ModelCapabilityProfileV2,
    ModelHealthSnapshotV2,
    ModelPriceScheduleV2,
    ModelQualityBaselineV2,
    ModelRouteCandidateV2,
    ModelRouteDecisionV2,
    ModelRouteRejectionCodeV2,
    ModelRouteRequestV2,
    ModelRoutingPolicyV2,
    PromptTemplateV2,
    RAGRequestV2,
    RAGResultV2,
    RAGSourcePolicyV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding

HASH = "a" * 64


def _ref(object_type: str, suffix: str = "example") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/v2",
        object_version="v2",
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 6, tzinfo=UTC),
        created_by="ai-gateway-contract-test",
        governing_versions=(
            VersionBinding(
                component="graph-engineered-eval-factory",
                version="gateway-v1",
                sha256=HASH,
            ),
        ),
    )


def _profile() -> ModelCapabilityProfileV2:
    return ModelCapabilityProfileV2.create(
        model_profile_id="model-profile://planner-v1",
        provider_id="provider://offline",
        model_id="model://planner",
        model_version="2026-08",
        capabilities=("long-context", "planning", "reasoning", "structured-output"),
        supported_data_classifications=("INTERNAL_DERIVED",),
        supported_residencies=("LOCAL",),
        context_limit_tokens=200_000,
        output_limit_tokens=16_000,
        availability="AVAILABLE",
        audit=_audit(),
    )


def _prompt() -> PromptTemplateV2:
    return PromptTemplateV2.create(
        prompt_template_id="prompt-template://requirement-planner/v1",
        agent_role="requirement-planner",
        task_kind="planning",
        system_template_ref=_ref("prompt-template-content", "system"),
        instruction_template_ref=_ref("prompt-template-content", "instruction"),
        required_input_object_types=("evaluation-requirement-spec",),
        output_schema_ref=_ref("json-schema", "dataset-build-plan"),
        allowed_tool_ids=("requirement-query",),
        required_model_capabilities=("planning", "structured-output"),
        injection_policy_ref=_ref("prompt-injection-policy"),
        template_version=1,
        audit=_audit(),
    )


def _route_request(
    *,
    route_version: int = 1,
    predecessor_route_ref: ObjectRef | None = None,
    failed_receipt_ref: ObjectRef | None = None,
) -> ModelRouteRequestV2:
    return ModelRouteRequestV2.create(
        route_request_id=f"model-route-request://planning/{route_version}",
        agent_task_ref=_ref("agent-task"),
        agent_definition_ref=_ref("agent-definition"),
        task_kind="planning",
        prompt_template_ref=_prompt().to_ref(),
        required_capabilities=("planning", "structured-output"),
        data_classification="INTERNAL_DERIVED",
        residency="LOCAL",
        input_token_budget=10_000,
        output_token_budget=4_000,
        max_cost_micro_usd=1_000_000,
        minimum_quality_basis_points=8500,
        allowed_model_profile_refs=(_profile().to_ref(),),
        budget_reservation_ref=_ref("work-model-reservation"),
        route_version=route_version,
        predecessor_route_ref=predecessor_route_ref,
        failed_receipt_ref=failed_receipt_ref,
        generator_model_profile_ref=None,
        audit=_audit(),
    )


def test_profile_and_prompt_are_strict_frozen_and_content_free() -> None:
    profile = _profile()
    prompt = _prompt()

    assert profile.to_ref().object_type == "model-capability-profile"
    assert prompt.to_ref().object_type == "prompt-template"
    with pytest.raises(ValidationError):
        profile.context_limit_tokens = 1  # type: ignore[misc]

    payload = prompt.model_dump(mode="python")
    payload["rendered_prompt"] = "forbidden"
    with pytest.raises(ValidationError):
        PromptTemplateV2.model_validate(payload)


def test_route_successor_requires_predecessor_and_failure_receipt() -> None:
    _route_request()
    with pytest.raises(ValidationError, match="successor route"):
        _route_request(route_version=2)
    with pytest.raises(ValidationError, match="first route"):
        _route_request(
            route_version=1,
            predecessor_route_ref=_ref("model-route-decision", "prior"),
            failed_receipt_ref=_ref("gateway-receipt"),
        )

    successor = _route_request(
        route_version=2,
        predecessor_route_ref=_ref("model-route-decision", "prior"),
        failed_receipt_ref=_ref("gateway-receipt"),
    )
    assert successor.predecessor_route_ref == _ref("model-route-decision", "prior")


def test_route_decision_requires_selected_candidate_and_ordered_rejections() -> None:
    request = _route_request()
    selected = ModelRouteCandidateV2(
        model_profile_ref=_profile().to_ref(),
        eligible=True,
        rejection_codes=(),
        quality_basis_points=9500,
        health_basis_points=9900,
        latency_milliseconds=500,
        estimated_cost_micro_usd=100_000,
        score=9_000_000,
    )
    rejected = ModelRouteCandidateV2(
        model_profile_ref=_ref("model-capability-profile", "rejected"),
        eligible=False,
        rejection_codes=(ModelRouteRejectionCodeV2.GOVERNANCE_DATA_CLASSIFICATION,),
        quality_basis_points=None,
        health_basis_points=None,
        latency_milliseconds=None,
        estimated_cost_micro_usd=None,
        score=None,
    )
    decision = ModelRouteDecisionV2.create(
        route_decision_id="model-route-decision://planning/1",
        request_ref=request.to_ref(),
        route_version=1,
        predecessor_route_ref=None,
        selected_model_profile_ref=_profile().to_ref(),
        candidates=(selected, rejected),
        routing_policy_ref=_ref("model-routing-policy"),
        quality_baseline_ref=_ref("model-quality-baseline"),
        health_snapshot_ref=_ref("model-health-snapshot"),
        price_schedule_ref=_ref("model-price-schedule"),
        budget_reservation_ref=_ref("work-model-reservation"),
        audit=_audit(),
    )
    assert decision.selected_model_profile_ref == selected.model_profile_ref

    payload = decision.model_dump(mode="python")
    payload["selected_model_profile_ref"] = rejected.model_profile_ref
    with pytest.raises(ValidationError, match="eligible candidate"):
        ModelRouteDecisionV2.create(
            route_decision_id="model-route-decision://invalid",
            request_ref=request.to_ref(),
            route_version=1,
            predecessor_route_ref=None,
            selected_model_profile_ref=rejected.model_profile_ref,
            candidates=(selected, rejected),
            routing_policy_ref=_ref("model-routing-policy"),
            quality_baseline_ref=_ref("model-quality-baseline"),
            health_snapshot_ref=_ref("model-health-snapshot"),
            price_schedule_ref=_ref("model-price-schedule"),
            budget_reservation_ref=_ref("work-model-reservation"),
            audit=_audit(),
        )


def test_gateway_receipt_and_result_bind_route_usage_and_private_body_ref() -> None:
    request = GatewayInvocationRequestV2.create(
        invocation_request_id="gateway-invocation-request://example",
        agent_task_ref=_ref("agent-task"),
        route_decision_ref=_ref("model-route-decision"),
        prompt_template_ref=_prompt().to_ref(),
        prompt_rendering_ref=_ref("prompt-rendering"),
        output_schema_ref=_ref("json-schema", "dataset-build-plan"),
        rag_result_refs=(),
        idempotency_key="gateway-invocation-example",
        audit=_audit(),
    )
    usage = GatewayUsageV2(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        charged_tokens=150,
        reported_cost_micro_usd=500,
    )
    receipt = GatewayReceiptV2.create(
        receipt_id="gateway-receipt://example",
        invocation_request_ref=request.to_ref(),
        route_decision_ref=request.route_decision_ref,
        status=GatewayInvocationStatusV2.SUCCEEDED,
        response_body_ref=_ref("model-response-content"),
        usage=usage,
        failure_code=None,
        completed_at=datetime(2026, 8, 6, tzinfo=UTC),
        audit=_audit(),
    )
    result = GatewayInvocationResultV2.create(
        invocation_result_id="gateway-invocation-result://example",
        invocation_request_ref=request.to_ref(),
        route_decision_ref=request.route_decision_ref,
        receipt_ref=receipt.to_ref(),
        status=GatewayInvocationStatusV2.SUCCEEDED,
        output_ref=_ref("dataset-build-plan"),
        failure_code=None,
        audit=_audit(),
    )
    assert result.receipt_ref == receipt.to_ref()
    assert not hasattr(receipt, "response_body")

    with pytest.raises(ValidationError, match="failed receipt"):
        GatewayReceiptV2.create(
            receipt_id="gateway-receipt://invalid",
            invocation_request_ref=request.to_ref(),
            route_decision_ref=request.route_decision_ref,
            status=GatewayInvocationStatusV2.FAILED,
            response_body_ref=_ref("model-response-content"),
            usage=usage,
            failure_code="MODEL_CALL_FAILED",
            completed_at=datetime(2026, 8, 6, tzinfo=UTC),
            audit=_audit(),
        )


def test_rag_contract_is_purpose_bound_and_contains_refs_not_text() -> None:
    policy = RAGSourcePolicyV2.create(
        source_policy_id="rag-source-policy://planner",
        allowed_principal_ids=("principal://planner",),
        allowed_purposes=("requirement-planning",),
        allowed_source_classes=("INTERNAL_SPEC",),
        projection_policy_ref=_ref("projection-policy"),
        max_results=5,
        max_bytes=50_000,
        max_characters=40_000,
        audit=_audit(),
    )
    request = RAGRequestV2.create(
        rag_request_id="rag-request://planner/example",
        agent_task_ref=_ref("agent-task"),
        principal_id="principal://planner",
        purpose="requirement-planning",
        query_ref=_ref("rag-query-content"),
        query_sha256=HASH,
        source_policy_ref=policy.to_ref(),
        requested_source_classes=("INTERNAL_SPEC",),
        max_results=3,
        max_bytes=20_000,
        max_characters=15_000,
        audit=_audit(),
    )
    result = RAGResultV2.create(
        rag_result_id="rag-result://planner/example",
        request_ref=request.to_ref(),
        approved_content_refs=(_ref("rag-content"),),
        provenance_refs=(_ref("provenance-decision"),),
        excluded_reason_codes=(),
        result_count=1,
        total_bytes=100,
        total_characters=90,
        audit=_audit(),
    )
    assert result.result_count == len(result.approved_content_refs)
    assert not hasattr(result, "retrieved_text")


def test_supporting_route_facts_are_versioned_and_bounded() -> None:
    profile = _profile()
    baseline = ModelQualityBaselineV2.create(
        baseline_id="model-quality-baseline://planner",
        model_profile_ref=profile.to_ref(),
        task_kind="planning",
        benchmark_ref=_ref("model-quality-benchmark"),
        quality_basis_points=9500,
        minimum_sample_count=100,
        audit=_audit(),
    )
    health = ModelHealthSnapshotV2.create(
        health_snapshot_id="model-health-snapshot://planner",
        model_profile_ref=profile.to_ref(),
        observed_at=datetime(2026, 8, 6, tzinfo=UTC),
        availability="HEALTHY",
        success_basis_points=9900,
        latency_milliseconds=500,
        audit=_audit(),
    )
    price = ModelPriceScheduleV2.create(
        price_schedule_id="model-price-schedule://planner",
        model_profile_ref=profile.to_ref(),
        input_micro_usd_per_million_tokens=1_000_000,
        output_micro_usd_per_million_tokens=2_000_000,
        effective_from=datetime(2026, 8, 6, tzinfo=UTC),
        audit=_audit(),
    )
    routing = ModelRoutingPolicyV2.create(
        routing_policy_id="model-routing-policy://core",
        allowed_provider_ids=("provider://offline",),
        allowed_agent_definition_refs=(_ref("agent-definition"),),
        allowed_model_profile_refs=(profile.to_ref(),),
        quality_weight=100,
        health_weight=10,
        latency_weight=1,
        cost_weight=1,
        audit=_audit(),
    )
    assert baseline.quality_basis_points == 9500
    assert health.success_basis_points == 9900
    assert price.output_micro_usd_per_million_tokens == 2_000_000
    assert routing.allowed_model_profile_refs == (profile.to_ref(),)


def test_gateway_contract_models_exclude_sensitive_bodies_and_credentials() -> None:
    forbidden = {
        "api_key",
        "credential",
        "prompt_body",
        "query_text",
        "rag_text",
        "response_body",
        "retrieved_text",
        "secret",
    }
    model_types = [
        value
        for _, value in inspect.getmembers(contracts, inspect.isclass)
        if issubclass(value, BaseModel) and value.__module__ == contracts.__name__
    ]
    assert model_types
    for model_type in model_types:
        assert forbidden.isdisjoint(model_type.model_fields), model_type.__name__
