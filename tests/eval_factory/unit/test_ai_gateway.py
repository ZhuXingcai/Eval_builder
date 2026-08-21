from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.ai_gateway.invocation import (
    EmbeddedAIGateway,
    ProviderInvocationResult,
)
from eval_factory.ai_gateway.model_catalog import ModelCatalog
from eval_factory.ai_gateway.policy import GatewayPolicyError, GatewaySecurityPolicy
from eval_factory.ai_gateway.prompt_registry import PromptRegistry
from eval_factory.ai_gateway.rag import (
    PurposeBoundRAGGateway,
    RAGAuthorizationError,
    RAGRetrievedDocument,
)
from eval_factory.ai_gateway.receipts import (
    GatewayRecordConflictError,
    GatewayRecordIntegrityError,
    GatewayRecordStore,
)
from eval_factory.ai_gateway.routing import (
    ModelRouteBlockedError,
    ModelRouteError,
    ModelRouter,
)
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationRequestV2,
    GatewayInvocationStatusV2,
    GatewayUsageV2,
    ModelCapabilityProfileV2,
    ModelHealthSnapshotV2,
    ModelPriceScheduleV2,
    ModelQualityBaselineV2,
    ModelRouteRejectionCodeV2,
    ModelRouteRequestV2,
    ModelRoutingPolicyV2,
    PromptTemplateV2,
    RAGRequestV2,
    RAGSourcePolicyV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding

HASH = "a" * 64
NOW = datetime(2026, 8, 6, tzinfo=UTC)


def _ref(object_type: str, suffix: str = "example") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/v2",
        object_version="v2",
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="ai-gateway-test",
        governing_versions=(
            VersionBinding(
                component="graph-engineered-eval-factory",
                version="gateway-v1",
                sha256=HASH,
            ),
        ),
    )


def _profile(name: str, capabilities: tuple[str, ...]) -> ModelCapabilityProfileV2:
    return ModelCapabilityProfileV2.create(
        model_profile_id=f"model-profile://{name}",
        provider_id="provider://offline",
        model_id=f"model://{name}",
        model_version="2026-08",
        capabilities=tuple(sorted(capabilities)),
        supported_data_classifications=("INTERNAL_DERIVED",),
        supported_residencies=("LOCAL",),
        context_limit_tokens=200_000,
        output_limit_tokens=16_000,
        availability="AVAILABLE",
        audit=_audit(),
    )


def _prompt(task_kind: str, capability: str) -> PromptTemplateV2:
    return PromptTemplateV2.create(
        prompt_template_id=f"prompt-template://{task_kind}/v1",
        agent_role=f"{task_kind}-agent",
        task_kind=task_kind,
        system_template_ref=_ref("prompt-template-content", f"{task_kind}-system"),
        instruction_template_ref=_ref(
            "prompt-template-content",
            f"{task_kind}-instruction",
        ),
        required_input_object_types=("evaluation-requirement-spec",),
        output_schema_ref=_ref("json-schema", task_kind),
        allowed_tool_ids=(),
        required_model_capabilities=(capability, "structured-output"),
        injection_policy_ref=_ref("prompt-injection-policy"),
        template_version=1,
        audit=_audit(),
    )


def _catalog() -> tuple[
    ModelCatalog,
    ModelRoutingPolicyV2,
    dict[str, ModelCapabilityProfileV2],
    dict[str, PromptTemplateV2],
]:
    profiles = {
        "planning": _profile(
            "planning",
            ("planning", "reasoning", "structured-output"),
        ),
        "extraction": _profile(
            "extraction",
            ("extraction", "structured-output", "text"),
        ),
        "rewrite": _profile(
            "rewrite",
            ("planning", "reasoning", "rewrite", "structured-output", "text"),
        ),
    }
    prompts = {name: _prompt(name, name) for name in profiles}
    baselines = [
        ModelQualityBaselineV2.create(
            baseline_id=f"model-quality-baseline://{name}",
            model_profile_ref=profile.to_ref(),
            task_kind=name,
            benchmark_ref=_ref("model-quality-benchmark", name),
            quality_basis_points=9200,
            minimum_sample_count=100,
            audit=_audit(),
        )
        for name, profile in profiles.items()
    ]
    baselines.append(
        ModelQualityBaselineV2.create(
            baseline_id="model-quality-baseline://rewrite-planning-backup",
            model_profile_ref=profiles["rewrite"].to_ref(),
            task_kind="planning",
            benchmark_ref=_ref("model-quality-benchmark", "rewrite-planning-backup"),
            quality_basis_points=8800,
            minimum_sample_count=100,
            audit=_audit(),
        )
    )
    health = tuple(
        ModelHealthSnapshotV2.create(
            health_snapshot_id=f"model-health-snapshot://{name}",
            model_profile_ref=profile.to_ref(),
            observed_at=NOW,
            availability="HEALTHY",
            success_basis_points=9900,
            latency_milliseconds=500,
            audit=_audit(),
        )
        for name, profile in profiles.items()
    )
    prices = tuple(
        ModelPriceScheduleV2.create(
            price_schedule_id=f"model-price-schedule://{name}",
            model_profile_ref=profile.to_ref(),
            input_micro_usd_per_million_tokens=1_000_000,
            output_micro_usd_per_million_tokens=2_000_000,
            effective_from=NOW,
            audit=_audit(),
        )
        for name, profile in profiles.items()
    )
    profile_refs = tuple(sorted((profile.to_ref() for profile in profiles.values()), key=_ref_key))
    agent_refs = tuple(
        sorted(
            (_ref("agent-definition", task_kind) for task_kind in profiles),
            key=_ref_key,
        )
    )
    policy = ModelRoutingPolicyV2.create(
        routing_policy_id="model-routing-policy://core",
        allowed_provider_ids=("provider://offline",),
        allowed_agent_definition_refs=agent_refs,
        allowed_model_profile_refs=profile_refs,
        quality_weight=100,
        health_weight=10,
        latency_weight=1,
        cost_weight=1,
        audit=_audit(),
    )
    return (
        ModelCatalog(
            profiles=tuple(profiles.values()),
            quality_baselines=tuple(baselines),
            health_snapshots=health,
            price_schedules=prices,
        ),
        policy,
        profiles,
        prompts,
    )


def _route_request(
    task_kind: str,
    profiles: dict[str, ModelCapabilityProfileV2],
    prompts: dict[str, PromptTemplateV2],
    *,
    route_version: int = 1,
    predecessor_route_ref: ObjectRef | None = None,
    failed_receipt_ref: ObjectRef | None = None,
    required_capabilities: tuple[str, ...] | None = None,
    max_cost_micro_usd: int = 1_000_000,
) -> ModelRouteRequestV2:
    refs = tuple(sorted((profile.to_ref() for profile in profiles.values()), key=_ref_key))
    return ModelRouteRequestV2.create(
        route_request_id=f"model-route-request://{task_kind}/{route_version}",
        agent_task_ref=_ref("agent-task", task_kind),
        agent_definition_ref=_ref("agent-definition", task_kind),
        task_kind=task_kind,
        prompt_template_ref=prompts[task_kind].to_ref(),
        required_capabilities=required_capabilities or (task_kind, "structured-output"),
        data_classification="INTERNAL_DERIVED",
        residency="LOCAL",
        input_token_budget=10_000,
        output_token_budget=4_000,
        max_cost_micro_usd=max_cost_micro_usd,
        minimum_quality_basis_points=8500,
        allowed_model_profile_refs=refs,
        budget_reservation_ref=_ref("work-model-reservation", task_kind),
        route_version=route_version,
        predecessor_route_ref=predecessor_route_ref,
        failed_receipt_ref=failed_receipt_ref,
        generator_model_profile_ref=None,
        audit=_audit(),
    )


def test_router_selects_distinct_models_for_three_task_families() -> None:
    catalog, policy, profiles, prompts = _catalog()
    router = ModelRouter(catalog, policy)

    decisions = {
        task_kind: router.route(_route_request(task_kind, profiles, prompts))
        for task_kind in ("planning", "extraction", "rewrite")
    }

    assert decisions["planning"].selected_model_profile_ref == profiles["planning"].to_ref()
    assert decisions["extraction"].selected_model_profile_ref == profiles["extraction"].to_ref()
    assert decisions["rewrite"].selected_model_profile_ref == profiles["rewrite"].to_ref()
    assert len({item.selected_model_profile_ref for item in decisions.values()}) == 3


def test_router_applies_governance_before_capability_and_quality() -> None:
    _, policy, _, prompts = _catalog()
    restricted = _profile("restricted", ("planning", "structured-output"))
    restricted = ModelCapabilityProfileV2.create(
        model_profile_id="model-profile://restricted",
        provider_id="provider://offline",
        model_id="model://restricted",
        model_version="2026-08",
        capabilities=("planning", "structured-output"),
        supported_data_classifications=("PUBLIC",),
        supported_residencies=("LOCAL",),
        context_limit_tokens=200_000,
        output_limit_tokens=16_000,
        availability="AVAILABLE",
        audit=_audit(),
    )
    profiles_only = {"planning": restricted}
    policy = ModelRoutingPolicyV2.create(
        routing_policy_id="model-routing-policy://restricted",
        allowed_provider_ids=("provider://offline",),
        allowed_agent_definition_refs=(_ref("agent-definition", "planning"),),
        allowed_model_profile_refs=(restricted.to_ref(),),
        quality_weight=1,
        health_weight=1,
        latency_weight=1,
        cost_weight=1,
        audit=_audit(),
    )
    request = _route_request(
        "planning",
        profiles_only,
        {"planning": prompts["planning"]},
    )
    router = ModelRouter(
        ModelCatalog(
            profiles=(restricted,),
            quality_baselines=(),
            health_snapshots=(),
            price_schedules=(),
        ),
        policy,
    )

    with pytest.raises(ModelRouteBlockedError) as error:
        router.route(request)
    assert error.value.candidates[0].rejection_codes == (
        ModelRouteRejectionCodeV2.GOVERNANCE_DATA_CLASSIFICATION,
    )


def test_router_blocks_capability_quality_and_invalid_predecessor_paths() -> None:
    catalog, policy, profiles, prompts = _catalog()
    router = ModelRouter(catalog, policy)
    request = _route_request("planning", profiles, prompts)
    incompatible = _route_request(
        "planning",
        profiles,
        prompts,
        required_capabilities=("multimodal", "structured-output"),
    )
    with pytest.raises(ModelRouteBlockedError) as capability_error:
        router.route(incompatible)
    assert {candidate.rejection_codes[0] for candidate in capability_error.value.candidates} == {
        ModelRouteRejectionCodeV2.CAPABILITY_MISMATCH
    }

    planning = profiles["planning"]
    missing_quality_router = ModelRouter(
        ModelCatalog(
            profiles=(planning,),
            quality_baselines=(),
            health_snapshots=(),
            price_schedules=(),
        ),
        ModelRoutingPolicyV2.create(
            routing_policy_id="model-routing-policy://missing-quality",
            allowed_provider_ids=("provider://offline",),
            allowed_agent_definition_refs=(_ref("agent-definition", "planning"),),
            allowed_model_profile_refs=(planning.to_ref(),),
            quality_weight=1,
            health_weight=1,
            latency_weight=1,
            cost_weight=1,
            audit=_audit(),
        ),
    )
    single_profile_request = _route_request(
        "planning",
        {"planning": planning},
        {"planning": prompts["planning"]},
    )
    with pytest.raises(ModelRouteBlockedError) as quality_error:
        missing_quality_router.route(single_profile_request)
    assert quality_error.value.candidates[0].rejection_codes == (
        ModelRouteRejectionCodeV2.QUALITY_BASELINE_MISSING,
    )

    decision = router.route(request)
    with pytest.raises(ModelRouteError, match="first route"):
        router.route(request, predecessor=decision)

    unauthorized_policy = ModelRoutingPolicyV2.create(
        routing_policy_id="model-routing-policy://unauthorized-agent",
        allowed_provider_ids=("provider://offline",),
        allowed_agent_definition_refs=(_ref("agent-definition", "other"),),
        allowed_model_profile_refs=policy.allowed_model_profile_refs,
        quality_weight=1,
        health_weight=1,
        latency_weight=1,
        cost_weight=1,
        audit=_audit(),
    )
    with pytest.raises(ModelRouteBlockedError) as agent_error:
        ModelRouter(catalog, unauthorized_policy).route(request)
    assert {candidate.rejection_codes[0] for candidate in agent_error.value.candidates} == {
        ModelRouteRejectionCodeV2.GOVERNANCE_AGENT_NOT_ALLOWED
    }

    no_budget = _route_request(
        "planning",
        profiles,
        prompts,
        max_cost_micro_usd=0,
    )
    with pytest.raises(ModelRouteBlockedError) as budget_error:
        router.route(no_budget)
    assert ModelRouteRejectionCodeV2.BUDGET_EXCEEDED in {
        candidate.rejection_codes[0] for candidate in budget_error.value.candidates
    }

    planning_baseline = ModelQualityBaselineV2.create(
        baseline_id="model-quality-baseline://planning-only",
        model_profile_ref=planning.to_ref(),
        task_kind="planning",
        benchmark_ref=_ref("model-quality-benchmark", "planning-only"),
        quality_basis_points=9200,
        minimum_sample_count=100,
        audit=_audit(),
    )
    missing_health_router = ModelRouter(
        ModelCatalog(
            profiles=(planning,),
            quality_baselines=(planning_baseline,),
            health_snapshots=(),
            price_schedules=(),
        ),
        ModelRoutingPolicyV2.create(
            routing_policy_id="model-routing-policy://missing-health",
            allowed_provider_ids=("provider://offline",),
            allowed_agent_definition_refs=(_ref("agent-definition", "planning"),),
            allowed_model_profile_refs=(planning.to_ref(),),
            quality_weight=1,
            health_weight=1,
            latency_weight=1,
            cost_weight=1,
            audit=_audit(),
        ),
    )
    with pytest.raises(ModelRouteBlockedError) as health_error:
        missing_health_router.route(single_profile_request)
    assert health_error.value.candidates[0].rejection_codes == (
        ModelRouteRejectionCodeV2.HEALTH_SNAPSHOT_MISSING,
    )


class _Provider:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    async def invoke(
        self,
        request: GatewayInvocationRequestV2,
        *,
        model_profile: ModelCapabilityProfileV2,
    ) -> ProviderInvocationResult:
        self.calls += 1
        if self.fail:
            return ProviderInvocationResult(
                status=GatewayInvocationStatusV2.FAILED,
                response_body_ref=None,
                output_ref=None,
                usage=GatewayUsageV2(
                    input_tokens=100,
                    output_tokens=0,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    charged_tokens=100,
                    reported_cost_micro_usd=500,
                ),
                failure_code="MODEL_CALL_FAILED",
            )
        return ProviderInvocationResult(
            status=GatewayInvocationStatusV2.SUCCEEDED,
            response_body_ref=_ref("model-response-content"),
            output_ref=_ref("dataset-build-plan"),
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


@pytest.mark.asyncio
async def test_gateway_persists_route_before_provider_and_replays_receipt(tmp_path: Path) -> None:
    catalog, policy, profiles, prompts = _catalog()
    store = GatewayRecordStore(tmp_path / "gateway.sqlite3")
    provider = _Provider()
    gateway = EmbeddedAIGateway(
        router=ModelRouter(catalog, policy),
        catalog=catalog,
        prompts=PromptRegistry(tuple(prompts.values())),
        records=store,
        provider=provider,
        clock=lambda: NOW,
    )
    route_request = _route_request("planning", profiles, prompts)
    route = gateway.route(route_request)
    assert gateway.route(route_request) == route
    assert provider.calls == 0
    assert gateway.get_route(route.to_ref()) == route

    invocation = GatewayInvocationRequestV2.create(
        invocation_request_id="gateway-invocation-request://planning",
        agent_task_ref=route_request.agent_task_ref,
        route_decision_ref=route.to_ref(),
        prompt_template_ref=prompts["planning"].to_ref(),
        prompt_rendering_ref=_ref("prompt-rendering"),
        output_schema_ref=prompts["planning"].output_schema_ref,
        rag_result_refs=(),
        idempotency_key="invoke-planning",
        audit=_audit(),
    )
    first = await gateway.invoke(invocation, route=route)
    replay = await gateway.invoke(invocation, route=route)

    assert first == replay
    assert first.status is GatewayInvocationStatusV2.SUCCEEDED
    assert provider.calls == 1
    receipt = store.get_receipt(first.receipt_ref)
    assert receipt.status is GatewayInvocationStatusV2.SUCCEEDED
    assert store.commit_invocation(invocation, receipt, first) == first
    other = GatewayInvocationRequestV2.create(
        invocation_request_id="gateway-invocation-request://other",
        agent_task_ref=invocation.agent_task_ref,
        route_decision_ref=invocation.route_decision_ref,
        prompt_template_ref=invocation.prompt_template_ref,
        prompt_rendering_ref=invocation.prompt_rendering_ref,
        output_schema_ref=invocation.output_schema_ref,
        rag_result_refs=(),
        idempotency_key="invoke-other",
        audit=_audit(),
    )
    with pytest.raises(GatewayRecordConflictError, match="closure refs"):
        store.commit_invocation(other, receipt, first)


@pytest.mark.asyncio
async def test_failed_receipt_is_required_before_successor_route(tmp_path: Path) -> None:
    catalog, policy, profiles, prompts = _catalog()
    store = GatewayRecordStore(tmp_path / "gateway.sqlite3")
    provider = _Provider(fail=True)
    gateway = EmbeddedAIGateway(
        router=ModelRouter(catalog, policy),
        catalog=catalog,
        prompts=PromptRegistry(tuple(prompts.values())),
        records=store,
        provider=provider,
        clock=lambda: NOW,
    )
    first_request = _route_request("planning", profiles, prompts)
    first_route = gateway.route(first_request)
    invocation = GatewayInvocationRequestV2.create(
        invocation_request_id="gateway-invocation-request://planning",
        agent_task_ref=first_request.agent_task_ref,
        route_decision_ref=first_route.to_ref(),
        prompt_template_ref=prompts["planning"].to_ref(),
        prompt_rendering_ref=_ref("prompt-rendering"),
        output_schema_ref=prompts["planning"].output_schema_ref,
        rag_result_refs=(),
        idempotency_key="invoke-planning",
        audit=_audit(),
    )
    failed = await gateway.invoke(invocation, route=first_route)
    assert failed.status is GatewayInvocationStatusV2.FAILED
    receipt = store.get_receipt(failed.receipt_ref)

    successor_request = _route_request(
        "planning",
        profiles,
        prompts,
        route_version=2,
        predecessor_route_ref=first_route.to_ref(),
        failed_receipt_ref=receipt.to_ref(),
    )
    successor = gateway.route(successor_request)
    rejected = {
        candidate.model_profile_ref: candidate.rejection_codes
        for candidate in successor.candidates
        if not candidate.eligible
    }
    assert rejected[first_route.selected_model_profile_ref] == (
        ModelRouteRejectionCodeV2.PREDECESSOR_MODEL_EXCLUDED,
    )
    assert successor.selected_model_profile_ref != first_route.selected_model_profile_ref


def test_prompt_registry_rejects_stale_or_incompatible_template() -> None:
    _, _, _, prompts = _catalog()
    registry = PromptRegistry(tuple(prompts.values()))
    assert registry.get("planning-agent", "planning") == prompts["planning"]
    with pytest.raises(ValueError, match="task kind"):
        registry.require_compatible(
            prompts["planning"].to_ref(),
            agent_role="planning-agent",
            task_kind="rewrite",
            required_capabilities=("planning", "structured-output"),
        )
    with pytest.raises(ValueError, match="unknown prompt"):
        registry.get_by_ref(_ref("prompt-template", "missing"))
    with pytest.raises(ValueError, match="unknown prompt role"):
        registry.get("missing-agent", "planning")
    with pytest.raises(ValueError, match="agent role"):
        registry.require_compatible(
            prompts["planning"].to_ref(),
            agent_role="other-agent",
            task_kind="planning",
            required_capabilities=("planning", "structured-output"),
        )
    with pytest.raises(ValueError, match="capabilities"):
        registry.require_compatible(
            prompts["planning"].to_ref(),
            agent_role="planning-agent",
            task_kind="planning",
            required_capabilities=("planning",),
        )
    with pytest.raises(ValueError, match="duplicate"):
        PromptRegistry((prompts["planning"], prompts["planning"]))


class _Retriever:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, request: RAGRequestV2) -> tuple[RAGRetrievedDocument, ...]:
        self.calls += 1
        return (
            RAGRetrievedDocument(
                content_ref=_ref("rag-content"),
                provenance_ref=_ref("provenance-decision"),
                source_class="INTERNAL_SPEC",
                size_bytes=100,
                character_count=90,
            ),
        )


def test_rag_document_and_unknown_policy_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rag-content"):
        RAGRetrievedDocument(
            content_ref=_ref("other-content"),
            provenance_ref=_ref("provenance-decision"),
            source_class="INTERNAL_SPEC",
            size_bytes=1,
            character_count=1,
        )
    with pytest.raises(ValueError, match="provenance"):
        RAGRetrievedDocument(
            content_ref=_ref("rag-content"),
            provenance_ref=_ref("other-provenance"),
            source_class="INTERNAL_SPEC",
            size_bytes=1,
            character_count=1,
        )
    with pytest.raises(ValueError, match="non-negative"):
        RAGRetrievedDocument(
            content_ref=_ref("rag-content"),
            provenance_ref=_ref("provenance-decision"),
            source_class="INTERNAL_SPEC",
            size_bytes=-1,
            character_count=1,
        )
    gateway = PurposeBoundRAGGateway(
        policies=(),
        retriever=_Retriever(),
        records=GatewayRecordStore(tmp_path / "unknown-rag.sqlite3"),
    )
    request = RAGRequestV2.create(
        rag_request_id="rag-request://unknown-policy",
        agent_task_ref=_ref("agent-task"),
        principal_id="principal://planner",
        purpose="requirement-planning",
        query_ref=_ref("rag-query-content"),
        query_sha256=HASH,
        source_policy_ref=_ref("rag-source-policy", "missing"),
        requested_source_classes=("INTERNAL_SPEC",),
        max_results=1,
        max_bytes=100,
        max_characters=100,
        audit=_audit(),
    )
    with pytest.raises(RAGAuthorizationError, match="unknown"):
        gateway.retrieve(request)


def test_rag_authorization_and_exact_replay_do_not_expose_or_refetch_text(
    tmp_path: Path,
) -> None:
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
    retriever = _Retriever()
    records = GatewayRecordStore(tmp_path / "rag.sqlite3")
    gateway = PurposeBoundRAGGateway(
        policies=(policy,),
        retriever=retriever,
        records=records,
    )
    request = RAGRequestV2.create(
        rag_request_id="rag-request://planner",
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
    first = gateway.retrieve(request)
    replay = gateway.retrieve(request)

    assert first == replay
    assert retriever.calls == 1
    assert not hasattr(first, "retrieved_text")
    assert records.commit_rag(request, first) == first

    denied = RAGRequestV2.create(
        rag_request_id="rag-request://denied",
        agent_task_ref=_ref("agent-task"),
        principal_id="principal://other",
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
    with pytest.raises(RAGAuthorizationError, match="principal"):
        gateway.retrieve(denied)
    assert retriever.calls == 1

    purpose_denied = RAGRequestV2.create(
        rag_request_id="rag-request://purpose-denied",
        agent_task_ref=_ref("agent-task"),
        principal_id="principal://planner",
        purpose="other-purpose",
        query_ref=_ref("rag-query-content"),
        query_sha256=HASH,
        source_policy_ref=policy.to_ref(),
        requested_source_classes=("INTERNAL_SPEC",),
        max_results=3,
        max_bytes=20_000,
        max_characters=15_000,
        audit=_audit(),
    )
    with pytest.raises(RAGAuthorizationError, match="purpose"):
        gateway.retrieve(purpose_denied)

    budget_denied = RAGRequestV2.create(
        rag_request_id="rag-request://budget-denied",
        agent_task_ref=_ref("agent-task"),
        principal_id="principal://planner",
        purpose="requirement-planning",
        query_ref=_ref("rag-query-content"),
        query_sha256=HASH,
        source_policy_ref=policy.to_ref(),
        requested_source_classes=("INTERNAL_SPEC",),
        max_results=6,
        max_bytes=20_000,
        max_characters=15_000,
        audit=_audit(),
    )
    with pytest.raises(RAGAuthorizationError, match="budget"):
        gateway.retrieve(budget_denied)

    with pytest.raises(ValueError, match="duplicate"):
        PurposeBoundRAGGateway(
            policies=(policy, policy),
            retriever=retriever,
            records=records,
        )


def test_gateway_policy_and_record_store_reject_missing_or_cross_bound_authority(
    tmp_path: Path,
) -> None:
    catalog, policy, profiles, prompts = _catalog()
    route_request = _route_request("planning", profiles, prompts)
    route = ModelRouter(catalog, policy).route(route_request)
    invocation = GatewayInvocationRequestV2.create(
        invocation_request_id="gateway-invocation-request://policy",
        agent_task_ref=route_request.agent_task_ref,
        route_decision_ref=route.to_ref(),
        prompt_template_ref=prompts["planning"].to_ref(),
        prompt_rendering_ref=_ref("prompt-rendering"),
        output_schema_ref=prompts["planning"].output_schema_ref,
        rag_result_refs=(),
        idempotency_key="invoke-policy",
        audit=_audit(),
    )
    GatewaySecurityPolicy.validate_invocation(
        invocation,
        route=route,
        prompt=prompts["planning"],
    )
    other_route = ModelRouter(catalog, policy).route(_route_request("extraction", profiles, prompts))
    with pytest.raises(GatewayPolicyError, match="persisted route"):
        GatewaySecurityPolicy.validate_invocation(
            invocation,
            route=other_route,
            prompt=prompts["planning"],
        )
    with pytest.raises(GatewayPolicyError, match="prompt template"):
        GatewaySecurityPolicy.validate_invocation(
            invocation,
            route=route,
            prompt=prompts["extraction"],
        )
    wrong_schema = GatewayInvocationRequestV2.create(
        invocation_request_id="gateway-invocation-request://wrong-schema",
        agent_task_ref=invocation.agent_task_ref,
        route_decision_ref=route.to_ref(),
        prompt_template_ref=prompts["planning"].to_ref(),
        prompt_rendering_ref=invocation.prompt_rendering_ref,
        output_schema_ref=_ref("json-schema", "wrong"),
        rag_result_refs=(),
        idempotency_key="invoke-wrong-schema",
        audit=_audit(),
    )
    with pytest.raises(GatewayPolicyError, match="output schema"):
        GatewaySecurityPolicy.validate_invocation(
            wrong_schema,
            route=route,
            prompt=prompts["planning"],
        )

    store = GatewayRecordStore(tmp_path / "records.sqlite3")
    assert store.get_invocation(invocation.to_ref()) is None
    assert store.get_rag(_ref("rag-request", "missing")) is None
    with pytest.raises(GatewayRecordIntegrityError, match="route decision"):
        store.get_route(route.to_ref())
    with pytest.raises(GatewayRecordIntegrityError, match="receipt"):
        store.get_receipt(_ref("gateway-receipt", "missing"))

    store.commit_route(route_request, route)
    with pytest.raises(GatewayRecordConflictError, match="does not bind"):
        store.commit_route(
            _route_request("extraction", profiles, prompts),
            route,
        )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE model_route_records
            SET decision_sha256 = ?
            WHERE decision_object_id = ?
            """,
            ("b" * 64, route.object_id),
        )
    with pytest.raises(GatewayRecordIntegrityError, match="materialized"):
        store.get_route(route.to_ref())


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (ref.object_type, ref.object_id, ref.object_version, ref.object_sha256)
