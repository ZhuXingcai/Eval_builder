from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.agent_system.dataset_graph_handlers import (
    FactoryDatasetGraphHandlers,
)
from eval_factory.agent_system.dataset_runtime import FactoryDatasetRuntime
from eval_factory.agent_system.graph import (
    FactoryControlGraph,
    FactoryGraphSignalV2,
)
from eval_factory.agent_system.plan_review import (
    PlanReviewDecisionSubmissionV1,
    PlanReviewResumeSubmissionV1,
    PlanReviewService,
)
from eval_factory.agent_system.planner import DatasetBuildPlanCompiler
from eval_factory.agent_system.private_store import FactoryPrivateObjectStore
from eval_factory.agent_system.registry import AgentRegistry
from eval_factory.agent_system.specialists import (
    GatewayRequirementPlannerAgent,
    RequirementPlannerConfig,
    RequirementPlanningInputV1,
    RequirementPlanningInputV2,
    RequirementPlanningMemoryProvider,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.ai_gateway.invocation import (
    EmbeddedAIGateway,
    ProviderInvocationResult,
)
from eval_factory.ai_gateway.model_catalog import ModelCatalog
from eval_factory.ai_gateway.prompt_registry import PromptRegistry
from eval_factory.ai_gateway.receipts import GatewayRecordStore
from eval_factory.ai_gateway.routing import ModelRouter
from eval_factory.contracts.agent_system_v2 import (
    AgentCapabilityV2,
    AgentDefinitionV2,
    DatasetBuildPlanTaskV2,
    DatasetBuildPlanV2,
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
    FactoryRunStatusV2,
    PlanDecisionKindV2,
    PlanKindV2,
)
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
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetNextActionV2,
    FactoryDatasetRunRequestV2,
)
from eval_factory.memory import (
    AgentMemoryService,
    AgentMemoryStore,
    MemoryAccessContextV1,
    MemoryCandidateV1,
    MemoryKindV1,
    MemoryNamespaceV1,
    MemorySensitivityV1,
    MemoryVisibilityV1,
)

HASH = "a" * 64
NOW = datetime(2026, 8, 7, tzinfo=UTC)
USER = "user://dataset-owner"


def _ref(object_type: str, suffix: str = "example") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/v2",
        object_version="v2",
        object_sha256=HASH,
    )


def _audit(actor: str = "factory-dataset-runtime-test") -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by=actor,
        governing_versions=(
            VersionBinding(
                component="factory-dataset-runtime",
                version="v1",
                sha256=HASH,
            ),
        ),
    )


def _requirement() -> EvaluationRequirementSpecV2:
    return EvaluationRequirementSpecV2.create(
        requirement_spec_id="evaluation-requirement-spec://runtime",
        run_id="factory-run://dataset/runtime",
        source_ref=_ref("evaluation-requirement-source"),
        goals=("Build safe source-grounded candidate items.",),
        constraints=("Do not expose restricted trace material.",),
        assumptions=("Provider boundary is deterministic in tests.",),
        open_questions=(),
        requirement_version=1,
        audit=_audit(),
    )


def _policy() -> FactoryRunPolicyV2:
    return FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://runtime",
        allowed_task_kinds=("task-rewrite", "trace-extraction"),
        max_transitions=512,
        max_plan_revisions=8,
        max_agent_attempts=2,
        max_model_requests=1_000,
        max_model_tokens=2_000_000,
        max_cost_micro_usd=25_000_000,
        audit=_audit(),
    )


def _request() -> FactoryDatasetRunRequestV2:
    return FactoryDatasetRunRequestV2.create(
        dataset_run_id=_requirement().run_id,
        requirement_spec_ref=_requirement().to_ref(),
        manifest_ref=_ref("trace-manifest"),
        source_authorization_ref=_ref("trace-source-authorization"),
        factory_policy_ref=_policy().to_ref(),
        pipeline_policy_refs=(
            _ref("batch-quality-policy"),
            _ref("release-projection-policy"),
        ),
        gateway_registry_refs=(
            _ref("agent-registry"),
            _ref("model-catalog"),
            _ref("prompt-registry"),
        ),
        output_target_ref=_ref("candidate-output-target"),
        idempotency_key="dataset-runtime-request",
        max_transitions=512,
        audit=_audit(),
    )


def _prompt() -> PromptTemplateV2:
    return PromptTemplateV2.create(
        prompt_template_id="prompt-template://planning/runtime",
        agent_role="planning-agent",
        task_kind="planning",
        system_template_ref=_ref(
            "prompt-template-content",
            "planning-system",
        ),
        instruction_template_ref=_ref(
            "prompt-template-content",
            "planning-instruction",
        ),
        required_input_object_types=("evaluation-requirement-spec",),
        output_schema_ref=_ref("json-schema", "planning"),
        allowed_tool_ids=(),
        required_model_capabilities=(
            "planning",
            "structured-output",
        ),
        injection_policy_ref=_ref("prompt-injection-policy"),
        template_version=1,
        audit=_audit(),
    )


def _profile() -> ModelCapabilityProfileV2:
    return ModelCapabilityProfileV2.create(
        model_profile_id="model-profile://planning/runtime",
        provider_id="provider://offline-runtime",
        model_id="model://planning/runtime",
        model_version="v1",
        capabilities=("planning", "structured-output"),
        supported_data_classifications=(
            "INTERNAL_DERIVED",
            "RESTRICTED_TRACE_DERIVED",
        ),
        supported_residencies=("LOCAL",),
        context_limit_tokens=200_000,
        output_limit_tokens=16_000,
        availability="AVAILABLE",
        audit=_audit(),
    )


def _registry() -> AgentRegistry:
    extract = AgentCapabilityV2.create(
        capability_id="agent-capability://trace-extraction",
        task_kinds=("trace-extraction",),
        input_object_types=("evaluation-requirement-spec",),
        output_object_types=("extracted-user-prompt",),
        model_capabilities=("structured-output",),
        tool_ids=("trace-query",),
        data_purposes=("evaluation-dataset-construction",),
        data_classifications=("RESTRICTED_TRACE_DERIVED",),
        audit=_audit(),
    )
    rewrite = AgentCapabilityV2.create(
        capability_id="agent-capability://task-rewrite",
        task_kinds=("task-rewrite",),
        input_object_types=("extracted-user-prompt",),
        output_object_types=("task-rewrite-candidate",),
        model_capabilities=("structured-output",),
        tool_ids=("task-rewrite",),
        data_purposes=("evaluation-dataset-construction",),
        data_classifications=("RESTRICTED_TRACE_DERIVED",),
        audit=_audit(),
    )
    definitions = tuple(
        AgentDefinitionV2.create(
            agent_definition_id=f"agent-definition://{role}",
            agent_role=role,
            agent_version="v1",
            capability_refs=(capability.to_ref(),),
            prompt_template_ref=_ref("prompt-template", role),
            model_policy_ref=_ref("model-routing-policy"),
            tool_ids=capability.tool_ids,
            data_purpose="evaluation-dataset-construction",
            allowed_data_classifications=("RESTRICTED_TRACE_DERIVED",),
            validator_refs=(_ref("validator", role),),
            max_attempts=2,
            max_model_requests=2,
            max_model_tokens=4_096,
            max_cost_micro_usd=100_000,
            workspace_isolated=True,
            network_allowed=False,
            audit=_audit(),
        )
        for role, capability in (
            ("trace-extraction-agent", extract),
            ("task-rewrite-agent", rewrite),
        )
    )
    return AgentRegistry(
        capabilities=(extract, rewrite),
        definitions=definitions,
    )


class _PlanningProvider:
    def __init__(
        self,
        private_store: FactoryPrivateObjectStore,
        *,
        memory_enabled: bool = False,
    ) -> None:
        self.private_store = private_store
        self.memory_enabled = memory_enabled
        self.calls = 0
        self.planning_inputs: list[RequirementPlanningInputV1 | RequirementPlanningInputV2] = []

    async def invoke(
        self,
        request: GatewayInvocationRequestV2,
        *,
        model_profile: ModelCapabilityProfileV2,
    ) -> ProviderInvocationResult:
        self.calls += 1
        assert "planning" in model_profile.capabilities
        planning = self.private_store.get_model(
            request.prompt_rendering_ref,
            RequirementPlanningInputV2 if self.memory_enabled else RequirementPlanningInputV1,
        )
        self.planning_inputs.append(planning)
        extract = DatasetBuildPlanTaskV2(
            task_key="extract",
            stage="core",
            task_kind="trace-extraction",
            agent_role="trace-extraction-agent",
            dependency_task_keys=(),
            input_object_types=("evaluation-requirement-spec",),
            output_object_types=("extracted-user-prompt",),
            required_capability_ids=("agent-capability://trace-extraction",),
            acceptance_check_refs=(_ref("acceptance-check", "extract"),),
            plan_review_kind=PlanKindV2.GLOBAL_BUILD,
            max_attempts=2,
            max_model_requests=1,
            max_model_tokens=4_096,
            max_cost_micro_usd=100_000,
        )
        rewrite = DatasetBuildPlanTaskV2(
            task_key="rewrite",
            stage="core",
            task_kind="task-rewrite",
            agent_role="task-rewrite-agent",
            dependency_task_keys=("extract",),
            input_object_types=("extracted-user-prompt",),
            output_object_types=("task-rewrite-candidate",),
            required_capability_ids=("agent-capability://task-rewrite",),
            acceptance_check_refs=(_ref("acceptance-check", "rewrite"),),
            plan_review_kind=PlanKindV2.TASK_REWRITE,
            max_attempts=2,
            max_model_requests=1,
            max_model_tokens=4_096,
            max_cost_micro_usd=100_000,
        )
        plan = DatasetBuildPlanV2.create(
            plan_id="dataset-build-plan://runtime/v1",
            run_ref=planning.run_ref,
            plan_version=1,
            predecessor_plan_ref=None,
            goals=planning.goals,
            user_constraints=planning.constraints,
            assumptions=planning.assumptions,
            unresolved_questions=planning.open_questions,
            stage_order=("core",),
            tasks=(rewrite, extract),
            required_review_kinds=(
                PlanKindV2.GLOBAL_BUILD,
                PlanKindV2.TASK_REWRITE,
            ),
            total_model_requests=2,
            total_model_tokens=8_192,
            total_cost_micro_usd=200_000,
            audit=_audit(),
        )
        output_ref = self.private_store.put_model(
            object_type="dataset-build-plan",
            value=plan,
        )
        return ProviderInvocationResult(
            status=GatewayInvocationStatusV2.SUCCEEDED,
            response_body_ref=self.private_store.put_text(
                object_type="model-response-content",
                text=plan.object_id,
            ),
            output_ref=output_ref,
            usage=GatewayUsageV2(
                input_tokens=100,
                output_tokens=50,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                charged_tokens=150,
                reported_cost_micro_usd=100,
            ),
            failure_code=None,
        )


def _runtime(
    tmp_path: Path,
    *,
    planning_memory_provider: RequirementPlanningMemoryProvider | None = None,
) -> tuple[
    FactoryDatasetRuntime,
    PlanReviewService,
    _PlanningProvider,
    FactoryControlStore,
]:
    private_store = FactoryPrivateObjectStore(tmp_path / "private")
    profile = _profile()
    prompt = _prompt()
    catalog = ModelCatalog(
        profiles=(profile,),
        quality_baselines=(
            ModelQualityBaselineV2.create(
                baseline_id="model-quality-baseline://planning/runtime",
                model_profile_ref=profile.to_ref(),
                task_kind="planning",
                benchmark_ref=_ref("model-quality-benchmark"),
                quality_basis_points=9_000,
                minimum_sample_count=100,
                audit=_audit(),
            ),
        ),
        health_snapshots=(
            ModelHealthSnapshotV2.create(
                health_snapshot_id="model-health-snapshot://planning/runtime",
                model_profile_ref=profile.to_ref(),
                observed_at=NOW,
                availability="HEALTHY",
                success_basis_points=9_900,
                latency_milliseconds=10,
                audit=_audit(),
            ),
        ),
        price_schedules=(
            ModelPriceScheduleV2.create(
                price_schedule_id="model-price-schedule://planning/runtime",
                model_profile_ref=profile.to_ref(),
                input_micro_usd_per_million_tokens=1,
                output_micro_usd_per_million_tokens=1,
                effective_from=NOW,
                audit=_audit(),
            ),
        ),
    )
    routing_policy = ModelRoutingPolicyV2.create(
        routing_policy_id="model-routing-policy://planning/runtime",
        allowed_provider_ids=("provider://offline-runtime",),
        allowed_agent_definition_refs=(_ref("agent-definition", "planning"),),
        allowed_model_profile_refs=(profile.to_ref(),),
        quality_weight=100,
        health_weight=10,
        latency_weight=1,
        cost_weight=1,
        audit=_audit(),
    )
    provider = _PlanningProvider(
        private_store,
        memory_enabled=planning_memory_provider is not None,
    )
    gateway = EmbeddedAIGateway(
        router=ModelRouter(catalog, routing_policy),
        catalog=catalog,
        prompts=PromptRegistry((prompt,)),
        records=GatewayRecordStore(tmp_path / "gateway.sqlite3"),
        provider=provider,
        clock=lambda: NOW,
    )
    planner = GatewayRequirementPlannerAgent(
        gateway=gateway,
        private_store=private_store,
        config=RequirementPlannerConfig(
            prompt=prompt,
            agent_definition_ref=_ref(
                "agent-definition",
                "planning",
            ),
            allowed_model_profile_refs=(profile.to_ref(),),
            budget_reservation_ref=_ref(
                "work-model-reservation",
                "planning",
            ),
        ),
        memory_provider=planning_memory_provider,
    )
    store = FactoryControlStore(tmp_path / "factory.sqlite3")
    compiler = DatasetBuildPlanCompiler(_registry())
    reviews = PlanReviewService(
        store,
        compiler=compiler,
        clock=lambda: NOW,
    )
    runtime = FactoryDatasetRuntime(
        store=store,
        planner=planner,
        compiler=compiler,
        plan_reviews=reviews,
        requested_by=USER,
    )
    return runtime, reviews, provider, store


@pytest.mark.asyncio
async def test_runtime_opens_global_review_and_replays_without_provider_call(
    tmp_path: Path,
) -> None:
    runtime, reviews, provider, store = _runtime(tmp_path)

    first = await runtime.advance(
        request=_request(),
        policy=_policy(),
        requirement=_requirement(),
        audit=_audit(),
    )
    replay = await runtime.advance(
        request=_request(),
        policy=_policy(),
        requirement=_requirement(),
        audit=_audit(),
    )

    assert first == replay
    assert first.status is FactoryRunStatusV2.WAITING_REVIEW
    assert first.next_action is FactoryDatasetNextActionV2.REVIEW_PLAN
    assert len(first.pending_review_refs) == 1
    assert provider.calls == 1
    assert store.get_dataset_request(_requirement().run_id) == _request()
    planning = store.get_planning_authority(_requirement().run_id)
    assert planning.plan_ref == store.get_plan(_requirement().run_id)[0].to_ref()

    review = reviews.show(first.pending_review_refs[0].object_id)
    assert review.request.plan_kind is PlanKindV2.GLOBAL_BUILD
    assert review.plan.goals == _requirement().goals


@pytest.mark.asyncio
async def test_requirement_planner_receives_authorized_cross_run_memory(
    tmp_path: Path,
) -> None:
    approval_ref = _ref("memory-admission-approval", "planning-memory")
    memory_access = MemoryAccessContextV1(
        tenant_id="tenant-a",
        project_id="project-a",
        subject_id="evaluation-planning",
        requester_agent_id="planning-agent",
    )
    memory_store = AgentMemoryStore(
        tmp_path / "memory.sqlite",
        trusted_approval_refs=frozenset({approval_ref}),
        trusted_access_contexts=frozenset({memory_access}),
    )
    candidate = MemoryCandidateV1(
        memory_id="agent-memory://evaluation-planning/source-grounding",
        namespace=MemoryNamespaceV1(
            tenant_id=memory_access.tenant_id,
            project_id=memory_access.project_id,
            subject_id=memory_access.subject_id,
            visibility=MemoryVisibilityV1.PROJECT_SHARED,
        ),
        kind=MemoryKindV1.EPISODIC,
        content=(
            "A past successful plan preserved source-grounded candidate items "
            "and explicit restricted-material constraints."
        ),
        tags=("planning", "source-grounding"),
        source_refs=(_ref("agent-result-envelope", "memory-source"),),
        valid_from=NOW,
        importance_basis_points=8_000,
        sensitivity=MemorySensitivityV1.RESTRICTED,
        approval_ref=approval_ref,
    )
    memory_record = memory_store.remember(
        candidate,
        access=memory_access,
        expected_revision=0,
        idempotency_key="planning-memory",
        created_at=NOW,
    )
    memory_store.remember(
        candidate.model_copy(
            update={
                "memory_id": "agent-memory://evaluation-planning/unapproved",
                "content": "An unapproved memory must not enter the Planner prompt.",
                "sensitivity": MemorySensitivityV1.INTERNAL,
                "approval_ref": None,
            }
        ),
        access=memory_access,
        expected_revision=0,
        idempotency_key="unapproved-planning-memory",
        created_at=NOW,
    )
    runtime, _reviews, provider, _store = _runtime(
        tmp_path,
        planning_memory_provider=RequirementPlanningMemoryProvider(
            service=AgentMemoryService(store=memory_store),
            access=memory_access,
        ),
    )

    await runtime.advance(
        request=_request(),
        policy=_policy(),
        requirement=_requirement(),
        audit=_audit(),
    )

    assert provider.calls == 1
    assert len(provider.planning_inputs) == 1
    planning = provider.planning_inputs[0]
    assert isinstance(planning, RequirementPlanningInputV2)
    assert len(planning.memory_context) == 1
    assert planning.memory_context[0].memory_record_ref == memory_record.to_ref()
    assert planning.memory_context[0].admission_approval_ref == candidate.approval_ref
    assert planning.memory_context[0].content == candidate.content


@pytest.mark.asyncio
async def test_runtime_continues_only_after_exact_review_resume(
    tmp_path: Path,
) -> None:
    runtime, reviews, provider, _store = _runtime(tmp_path)
    waiting = await runtime.advance(
        request=_request(),
        policy=_policy(),
        requirement=_requirement(),
        audit=_audit(),
    )
    review_id = waiting.pending_review_refs[0].object_id
    reviews.decide(
        review_id,
        PlanReviewDecisionSubmissionV1(
            expected_plan_version=1,
            decision=PlanDecisionKindV2.APPROVE,
            decided_by=USER,
            reason_code="GLOBAL_PLAN_APPROVED",
            idempotency_key="approve-global-runtime",
        ),
        audit=_audit(),
    )
    reviews.resume(
        review_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=1,
            resumed_by=USER,
            idempotency_key="resume-global-runtime",
        ),
        audit=_audit(),
    )

    resumed = await runtime.advance(
        request=_request(),
        policy=_policy(),
        requirement=_requirement(),
        audit=_audit(),
    )
    assert resumed.status is FactoryRunStatusV2.PLANNING
    assert resumed.pending_review_refs == ()
    assert resumed.next_action is FactoryDatasetNextActionV2.ADVANCE
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_runtime_rejects_request_value_drift_before_side_effect(
    tmp_path: Path,
) -> None:
    runtime, _reviews, provider, store = _runtime(tmp_path)
    changed_requirement = EvaluationRequirementSpecV2.create(
        requirement_spec_id=_requirement().requirement_spec_id,
        run_id=_requirement().run_id,
        source_ref=_requirement().source_ref,
        goals=("Changed goal.",),
        constraints=_requirement().constraints,
        assumptions=_requirement().assumptions,
        open_questions=(),
        requirement_version=1,
        audit=_audit(),
    )

    with pytest.raises(
        ValueError,
        match="request requirement authority",
    ):
        await runtime.advance(
            request=_request(),
            policy=_policy(),
            requirement=changed_requirement,
            audit=_audit(),
        )
    assert provider.calls == 0
    with pytest.raises(Exception, match="not found"):
        store.get_run(_requirement().run_id)


@pytest.mark.asyncio
async def test_dataset_graph_handlers_drive_runtime_without_body_state(
    tmp_path: Path,
) -> None:
    runtime, reviews, provider, store = _runtime(tmp_path)
    waiting = await runtime.advance(
        request=_request(),
        policy=_policy(),
        requirement=_requirement(),
        audit=_audit(),
    )
    handlers = FactoryDatasetGraphHandlers(
        runtime=runtime,
        request=_request(),
        policy=_policy(),
        requirement=_requirement(),
        core_input=None,
        audit=_audit(),
    )
    mapping = handlers.mapping()
    assert set(mapping) == {
        "requirement_planner",
        "review_global_plan",
        "compile_global_plan",
        "dispatch_ready_work",
        "validate_core_vertical",
        "planner_assessment",
        "assemble_core_output",
        "review_final_delivery",
        "export_core_output",
    }
    current = store.get_run(_requirement().run_id)
    graph = FactoryControlGraph(
        store,
        handlers=mapping,
    ).compile()
    pending = await graph.ainvoke(
        {
            "factory_run_id": current.run_id,
            "expected_run_version": current.run_version,
            "transition_count": 0,
            "signal": FactoryGraphSignalV2.CONTINUE,
        }
    )
    assert pending["current_node"] == "review_global_plan"
    assert pending["signal"] is FactoryGraphSignalV2.WAIT
    assert provider.calls == 1

    review_id = waiting.pending_review_refs[0].object_id
    reviews.decide(
        review_id,
        PlanReviewDecisionSubmissionV1(
            expected_plan_version=1,
            decision=PlanDecisionKindV2.APPROVE,
            decided_by=USER,
            reason_code="GLOBAL_PLAN_APPROVED",
            idempotency_key="approve-global-graph-runtime",
        ),
        audit=_audit(),
    )
    reviews.resume(
        review_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=1,
            resumed_by=USER,
            idempotency_key="resume-global-graph-runtime",
        ),
        audit=_audit(),
    )
    resumed = store.get_run(_requirement().run_id)
    advanced = await graph.ainvoke(
        {
            "factory_run_id": resumed.run_id,
            "expected_run_version": resumed.run_version,
            "transition_count": 0,
            "signal": FactoryGraphSignalV2.CONTINUE,
        }
    )

    assert advanced["current_node"] == "planner_assessment"
    assert advanced["signal"] is FactoryGraphSignalV2.WAIT
    assert provider.calls == 1
    assert set(advanced).isdisjoint(
        {
            "credential",
            "model_output_body",
            "prompt_body",
            "raw_trace",
            "runtime_transcript",
        }
    )
