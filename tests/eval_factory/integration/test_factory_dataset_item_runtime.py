from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from test_core_vertical_real_traces import (
    RAW_ROOT,
    _audit,
    _gateway,
    _ref,
    _ref_key,
)
from test_factory_dataset_core_runtime import _registry
from test_factory_task_authoring_bridge import (
    _one_trace_root,
    _profile,
    _prompt,
    _r4_agent,
)

from env_mock_agent.facade import (
    AttachmentExecutionRequestV2,
    AttachmentExecutionResultV2,
    AttachmentExecutionStatusV2,
    AttachmentRouteCandidateKind,
    AttachmentRouteDecisionV2,
    AttachmentRouteOutcome,
    AttachmentRouteRequestV2,
    ExecutionTelemetryV2,
    FacadeObjectRef,
    WorldLedgerSnapshotRequestV2,
    WorldLedgerSnapshotV2,
    attachment_execution_request_ref,
    attachment_execution_result_carried_sha256,
    attachment_route_decision_carried_sha256,
    attachment_route_request_ref,
    world_ledger_snapshot_carried_sha256,
    world_ledger_snapshot_request_ref,
)
from eval_factory.agent_system.attachment_item_runtime import (
    FactoryAttachmentItemRuntime,
    FactoryAttachmentItemState,
)
from eval_factory.agent_system.attachment_planner import (
    AttachmentPlanningAgentConfig,
    AttachmentPlanningInputV1,
    GatewayAttachmentPlanningAgent,
)
from eval_factory.agent_system.attachment_planning import (
    AttachmentGenerationPlanCompiler,
)
from eval_factory.agent_system.attachment_preparation import (
    AttachmentR5PreparationValidator,
)
from eval_factory.agent_system.attachment_preparation_builder import (
    FactoryAttachmentPreparationBuilder,
    FactoryAttachmentPreparationConfig,
)
from eval_factory.agent_system.attachment_preparation_material import (
    FactoryAttachmentPreparationMaterialError,
    FactoryAttachmentPreparationMaterialStore,
)
from eval_factory.agent_system.attachment_registry import (
    AttachmentAgentRegistryConfig,
    build_attachment_agent_registry,
)
from eval_factory.agent_system.attachment_subgraph import (
    SupervisedAttachmentR5Runner,
)
from eval_factory.agent_system.core_material import (
    FactoryCoreMaterialStore,
)
from eval_factory.agent_system.core_vertical import CoreVerticalRunner
from eval_factory.agent_system.dataset_runtime import (
    FactoryDatasetCoreInput,
    FactoryDatasetRuntime,
)
from eval_factory.agent_system.job_store_bridge import (
    FactoryJobStoreBridge,
    FactoryJobStoreTemplate,
)
from eval_factory.agent_system.plan_review import (
    PlanReviewDecisionSubmissionV1,
    PlanReviewResumeSubmissionV1,
    PlanReviewService,
)
from eval_factory.agent_system.planner import DatasetBuildPlanCompiler
from eval_factory.agent_system.specialists import (
    GatewayIntentRewriteAgent,
    GatewayRequirementPlannerAgent,
    RequirementPlannerConfig,
    SemanticAgentConfig,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.agent_system.supervisor import ExecutionSupervisor
from eval_factory.agent_system.task_authoring_bridge import (
    FactoryTaskAuthoringBridge,
)
from eval_factory.agent_system.task_authoring_material import (
    FactoryTaskAuthoringMaterialStore,
)
from eval_factory.agent_system.workspace import AgentWorkspaceManager
from eval_factory.ai_gateway.invocation import (
    EmbeddedAIGateway,
    ProviderInvocationResult,
)
from eval_factory.ai_gateway.model_catalog import ModelCatalog
from eval_factory.ai_gateway.prompt_registry import PromptRegistry
from eval_factory.ai_gateway.receipts import GatewayRecordStore
from eval_factory.ai_gateway.routing import ModelRouter
from eval_factory.approval.requests import (
    UserApprovalPolicyCompiler,
)
from eval_factory.contracts.agent_system_v2 import (
    AttachmentGenerationPlanV2,
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
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
)
from eval_factory.contracts.approval import (
    ApprovalMode,
    FinalReviewScope,
)
from eval_factory.contracts.attachment_v2 import (
    ArtifactRoutingPolicyV2,
    PublicSourceRetrievalPolicyV2,
    artifact_routing_policy_carried_sha256,
    public_source_retrieval_policy_carried_sha256,
)
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunRequestV2,
    FactoryItemStageV2,
)
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    ItemStatus,
    JobStatus,
    ResourceBudget,
)
from eval_factory.contracts.orchestration_v2 import (
    StageNameV2,
    dataset_item_id_v2,
)
from eval_factory.orchestration.job_store import JobStore

USER = "user://dataset-item-runtime-owner"

pytestmark = pytest.mark.skipif(
    not (RAW_ROOT / "manifest.csv").is_file(),
    reason="private 91-trace corpus is not installed",
)


def _requirement() -> EvaluationRequirementSpecV2:
    return EvaluationRequirementSpecV2.create(
        requirement_spec_id=("evaluation-requirement-spec://item-runtime"),
        run_id="factory-run://dataset/item-runtime",
        source_ref=_ref(
            "evaluation-requirement-source",
            "item-runtime",
        ),
        goals=("Evaluate source-grounded instruction following.",),
        constraints=("Do not expose original final output.",),
        assumptions=(),
        open_questions=(),
        requirement_version=1,
        audit=_audit(),
    )


def _policy() -> FactoryRunPolicyV2:
    return FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://item-runtime",
        allowed_task_kinds=(
            "attachment-mock",
            "task-authoring",
            "task-rewrite",
            "trace-extraction",
        ),
        max_transitions=512,
        max_plan_revisions=8,
        max_agent_attempts=2,
        max_model_requests=1_000,
        max_model_tokens=2_000_000,
        max_cost_micro_usd=25_000_000,
        audit=_audit(),
    )


def _job_template() -> FactoryJobStoreTemplate:
    return FactoryJobStoreTemplate(
        requested_stages=(
            StageNameV2.TRACE_INDEX,
            StageNameV2.SAFETY,
            StageNameV2.LABEL,
            StageNameV2.TASK_AUTHORING,
            StageNameV2.ATTACHMENT,
            StageNameV2.ITEM_QUALITY,
            StageNameV2.BATCH_QUALITY,
            StageNameV2.RELEASE,
        ),
        privacy_profile="trusted-monitored-local",
        model_profiles=(),
        budget=ResourceBudget(
            max_model_requests=100,
            max_model_tokens=1_000_000,
            max_processes=4,
            max_renderers=4,
            max_network_requests=100,
            max_storage_bytes=10_000_000,
        ),
        concurrency=ConcurrencyLimit(
            model_requests=2,
            processes=2,
            renderers=2,
            network_requests=2,
            artifacts_per_item=4,
            items=4,
        ),
        approval_policy=UserApprovalPolicyCompiler().compile(
            mode=ApprovalMode.NONE,
            custom_checkpoints=frozenset(),
            final_review_scope=FinalReviewScope.NONE,
            explicit_choice=True,
            audit=_audit(),
        ),
        export_target=ExportTarget(
            profile="LH",
            profile_version="v1",
            channel="CANARY",
            registry="registry://factory-candidate",
        ),
    )


def _request(
    manifest_sha256: str,
) -> FactoryDatasetRunRequestV2:
    manifest_ref = _ref(
        "trace-manifest",
        manifest_sha256,
    ).model_copy(
        update={
            "object_id": (f"trace-manifest://sha256/{manifest_sha256}"),
            "object_sha256": manifest_sha256,
        }
    )
    return FactoryDatasetRunRequestV2.create(
        dataset_run_id=_requirement().run_id,
        requirement_spec_ref=_requirement().to_ref(),
        manifest_ref=manifest_ref,
        source_authorization_ref=_ref(
            "trace-source-authorization",
            "item-runtime",
        ),
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
        idempotency_key="dataset-item-runtime",
        max_transitions=512,
        audit=_audit(),
    )


def _retrieval_policy() -> PublicSourceRetrievalPolicyV2:
    policy = PublicSourceRetrievalPolicyV2(
        public_source_retrieval_policy_id=("public-source-retrieval-policy://pending"),
        approved_search_provider_ids=(),
        approved_fetch_provider_ids=("fetch-provider://disabled",),
        allowed_schemes=("https",),
        allowed_host_suffixes=("example.gov",),
        max_search_results=1,
        max_fetch_bytes=1_024,
        query_egress_policy_ref=_ref(
            "query-egress-policy",
            "item-runtime",
        ),
        network_policy_ref=_ref(
            "network-policy",
            "item-runtime",
        ),
        source_usage_policy_ref=_ref(
            "source-usage-policy",
            "item-runtime",
        ),
        safety_scan_policy_ref=_ref(
            "safety-scan-policy",
            "item-runtime",
        ),
        license_policy_ref=_ref(
            "source-license-policy",
            "item-runtime",
        ),
        policy_version="public-source-retrieval/r5-04-v1",
        public_source_retrieval_policy_sha256="0" * 64,
        audit=_audit(),
    )
    digest = public_source_retrieval_policy_carried_sha256(policy)
    return policy.model_copy(
        update={
            "public_source_retrieval_policy_id": (f"public-source-retrieval-policy://sha256/{digest}"),
            "public_source_retrieval_policy_sha256": digest,
        }
    )


def _routing_policy() -> ArtifactRoutingPolicyV2:
    model_refs = (
        _ref("model-profile", "sdk"),
        _ref("model-profile", "cli"),
        _ref("model-profile", "pi"),
    )
    provider_policy_ref = _ref(
        "provider-capability-policy",
        "item-runtime",
    )
    runtime_policy_ref = _ref(
        "runtime-capability-policy",
        "item-runtime",
    )
    validator_policy_ref = _ref(
        "validator-policy",
        "item-runtime",
    )
    audit = _audit().model_copy(
        update={
            "input_refs": tuple(
                sorted(
                    (
                        *model_refs,
                        provider_policy_ref,
                        runtime_policy_ref,
                        validator_policy_ref,
                    ),
                    key=_ref_key,
                )
            )
        }
    )
    policy = ArtifactRoutingPolicyV2(
        artifact_routing_policy_id=("artifact-routing-policy://pending"),
        deterministic_provider_first=True,
        approved_provider_ids=("text",),
        runtime_order=(
            "claude_agent_sdk",
            "claude_code_cli",
            "pi_rpc",
        ),
        runtime_model_profile_refs=model_refs,
        provider_capability_policy_ref=provider_policy_ref,
        runtime_capability_policy_ref=runtime_policy_ref,
        validator_policy_ref=validator_policy_ref,
        silent_degradation_allowed=False,
        policy_version="artifact-routing/r5-05-v1",
        artifact_routing_policy_sha256="0" * 64,
        audit=audit,
    )
    digest = artifact_routing_policy_carried_sha256(policy)
    return policy.model_copy(
        update={
            "artifact_routing_policy_id": (f"artifact-routing-policy://sha256/{digest}"),
            "artifact_routing_policy_sha256": digest,
        }
    )


class _RoutingFacade:
    def __init__(self) -> None:
        self.calls = 0

    async def route(
        self,
        request: AttachmentRouteRequestV2,
    ) -> AttachmentRouteDecisionV2:
        self.calls += 1
        decision = AttachmentRouteDecisionV2(
            route_decision_id=("attachment-route-decision://pending"),
            route_request_ref=attachment_route_request_ref(request),
            outcome=(AttachmentRouteOutcome.SELECTED_PROVIDER),
            selected_kind=(AttachmentRouteCandidateKind.PROVIDER),
            selected_id="text",
            selected_version="fixture-v1",
            satisfied_capability_ids=(request.required_provider_capability_ids),
            skipped=(),
            capability_snapshot_sha256=(request.route_request_sha256),
            probed_at=_audit().created_at,
            policy_version="artifact-routing/r5-05-v1",
            route_decision_sha256="0" * 64,
        )
        digest = attachment_route_decision_carried_sha256(decision)
        return decision.model_copy(
            update={
                "route_decision_id": (f"attachment-route-decision://sha256/{digest}"),
                "route_decision_sha256": digest,
            }
        )


class _SnapshotFacade:
    def __init__(self) -> None:
        self.calls = 0
        self.execution_calls = 0

    async def snapshot_world(
        self,
        request: WorldLedgerSnapshotRequestV2,
    ) -> WorldLedgerSnapshotV2:
        self.calls += 1
        snapshot = WorldLedgerSnapshotV2(
            world_ledger_snapshot_id=("world-ledger-snapshot://pending"),
            snapshot_request_ref=(world_ledger_snapshot_request_ref(request)),
            world_ledger_ref=request.world_ledger_ref,
            fact_locks=(),
            ledger_sha256=(request.world_ledger_ref.object_sha256),
            policy_version="artifact-execution/r5-06-v1",
            snapshotted_at=_audit().created_at,
            world_ledger_snapshot_sha256="0" * 64,
        )
        digest = world_ledger_snapshot_carried_sha256(snapshot)
        return snapshot.model_copy(
            update={
                "world_ledger_snapshot_id": (f"world-ledger-snapshot://sha256/{digest}"),
                "world_ledger_snapshot_sha256": digest,
            }
        )

    async def execute(
        self,
        request: AttachmentExecutionRequestV2,
    ) -> AttachmentExecutionResultV2:
        self.execution_calls += 1
        output_sha256 = hashlib.sha256(request.artifact_id.encode()).hexdigest()
        result = AttachmentExecutionResultV2(
            execution_result_id=("attachment-execution-result://pending"),
            execution_request_ref=(attachment_execution_request_ref(request)),
            artifact_id=request.artifact_id,
            attempt=request.attempt,
            status=AttachmentExecutionStatusV2.SUCCEEDED,
            world_ledger_snapshot_ref=(request.world_ledger_snapshot_ref),
            selected_route_kind=request.selected_route_kind,
            selected_route_id=request.selected_route_id,
            worker_version="fixture-v1",
            output_ref=FacadeObjectRef(
                object_type="attachment-output",
                object_id=(f"attachment-output://sha256/{output_sha256}"),
                object_version="v2",
                object_sha256=output_sha256,
            ),
            output_sha256=output_sha256,
            retryable=False,
            failure_code=None,
            telemetry=ExecutionTelemetryV2.unavailable(observed_at=_audit().created_at),
            policy_version="artifact-execution/r5-06-v1",
            execution_result_sha256="0" * 64,
        )
        digest = attachment_execution_result_carried_sha256(result)
        return result.model_copy(
            update={
                "execution_result_id": (f"attachment-execution-result://sha256/{digest}"),
                "execution_result_sha256": digest,
            }
        )


class _AttachmentPlanningProvider:
    def __init__(self, private_store) -> None:
        self.private_store = private_store
        self.calls = 0

    async def invoke(
        self,
        request: GatewayInvocationRequestV2,
        *,
        model_profile: ModelCapabilityProfileV2,
    ) -> ProviderInvocationResult:
        self.calls += 1
        assert "attachment-planning" in (model_profile.capabilities)
        rendering = self.private_store.get_model(
            request.prompt_rendering_ref,
            AttachmentPlanningInputV1,
        )
        plan = AttachmentGenerationPlanV2.create(
            plan_id="attachment-generation-plan://item-runtime",
            run_ref=rendering.run_ref,
            plan_version=1,
            predecessor_plan_ref=None,
            producer_task_view_ref=(rendering.producer_task_view_ref),
            evidence_bundle_ref=rendering.evidence_bundle_ref,
            attachment_planning_context_ref=(rendering.attachment_planning_context_ref),
            works=rendering.work_templates,
            max_parallel_groups=len(rendering.work_templates),
            quality_policy_ref=rendering.quality_policy_ref,
            solvability_policy_ref=(rendering.solvability_policy_ref),
            total_model_requests=sum(value.max_model_requests for value in rendering.work_templates),
            total_model_tokens=sum(value.max_model_tokens for value in rendering.work_templates),
            total_cost_micro_usd=sum(value.max_cost_micro_usd for value in rendering.work_templates),
            audit=_audit(),
        )
        output_ref = self.private_store.put_model(
            object_type="attachment-generation-plan",
            value=plan,
        )
        return ProviderInvocationResult(
            status=GatewayInvocationStatusV2.SUCCEEDED,
            response_body_ref=self.private_store.put_text(
                object_type="model-response-content",
                text=output_ref.object_id,
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


def _attachment_planning_agent(
    tmp_path: Path,
    private_store,
) -> tuple[
    GatewayAttachmentPlanningAgent,
    _AttachmentPlanningProvider,
]:
    profile = _profile("attachment-planning")
    prompt = _prompt("attachment-planning")
    catalog = ModelCatalog(
        profiles=(profile,),
        quality_baselines=(
            ModelQualityBaselineV2.create(
                baseline_id=("model-quality-baseline://attachment-planning/item-runtime"),
                model_profile_ref=profile.to_ref(),
                task_kind="attachment-planning",
                benchmark_ref=_ref(
                    "model-quality-benchmark",
                    "attachment-planning",
                ),
                quality_basis_points=9_000,
                minimum_sample_count=100,
                audit=_audit(),
            ),
        ),
        health_snapshots=(
            ModelHealthSnapshotV2.create(
                health_snapshot_id=("model-health-snapshot://attachment-planning/item-runtime"),
                model_profile_ref=profile.to_ref(),
                observed_at=_audit().created_at,
                availability="HEALTHY",
                success_basis_points=9_900,
                latency_milliseconds=10,
                audit=_audit(),
            ),
        ),
        price_schedules=(
            ModelPriceScheduleV2.create(
                price_schedule_id=("model-price-schedule://attachment-planning/item-runtime"),
                model_profile_ref=profile.to_ref(),
                input_micro_usd_per_million_tokens=1,
                output_micro_usd_per_million_tokens=1,
                effective_from=_audit().created_at,
                audit=_audit(),
            ),
        ),
    )
    agent_ref = _ref(
        "agent-definition",
        "attachment-planning",
    )
    provider = _AttachmentPlanningProvider(private_store)
    gateway = EmbeddedAIGateway(
        router=ModelRouter(
            catalog,
            ModelRoutingPolicyV2.create(
                routing_policy_id=("model-routing-policy://attachment-planning/item-runtime"),
                allowed_provider_ids=("provider://offline-r4",),
                allowed_agent_definition_refs=(agent_ref,),
                allowed_model_profile_refs=(profile.to_ref(),),
                quality_weight=100,
                health_weight=10,
                latency_weight=1,
                cost_weight=1,
                audit=_audit(),
            ),
        ),
        catalog=catalog,
        prompts=PromptRegistry((prompt,)),
        records=GatewayRecordStore(tmp_path / "attachment-planning-gateway.sqlite3"),
        provider=provider,
    )
    registry = _attachment_registry()
    return (
        GatewayAttachmentPlanningAgent(
            gateway=gateway,
            private_store=private_store,
            compiler=AttachmentGenerationPlanCompiler(registry),
            config=AttachmentPlanningAgentConfig(
                prompt=prompt,
                agent_definition_ref=agent_ref,
                allowed_model_profile_refs=(profile.to_ref(),),
                budget_reservation_ref=_ref(
                    "work-model-reservation",
                    "attachment-planning",
                ),
            ),
        ),
        provider,
    )


def _attachment_registry():
    return build_attachment_agent_registry(
        config=AttachmentAgentRegistryConfig(
            mock_prompt_ref=_ref(
                "prompt-template",
                "attachment-mock",
            ),
            quality_prompt_ref=_ref(
                "prompt-template",
                "attachment-quality",
            ),
            solvability_prompt_ref=_ref(
                "prompt-template",
                "attachment-solvability",
            ),
            mock_model_policy_ref=_ref(
                "model-routing-policy",
                "attachment-mock",
            ),
            quality_model_policy_ref=_ref(
                "model-routing-policy",
                "attachment-quality",
            ),
            solvability_model_policy_ref=_ref(
                "model-routing-policy",
                "attachment-solvability",
            ),
        ),
        audit=_audit(),
    )


@pytest.mark.asyncio
async def test_runtime_promotes_complete_r4_authority_and_replays(
    tmp_path: Path,
) -> None:
    manifest, manifest_sha256 = _one_trace_root(tmp_path)
    gateway, private_store, core_provider, values = _gateway(tmp_path / "core")
    profiles = values["profiles"]
    prompts = values["prompts"]
    assert isinstance(profiles, dict)
    assert isinstance(prompts, dict)
    profile_refs = tuple(
        sorted(
            (profile.to_ref() for profile in profiles.values()),
            key=_ref_key,
        )
    )
    semantic_agent = GatewayIntentRewriteAgent(
        gateway=gateway,
        private_store=private_store,
        config=SemanticAgentConfig(
            intent_prompt=prompts["extraction"],
            rewrite_prompt=prompts["rewrite"],
            intent_agent_definition_ref=_ref(
                "agent-definition",
                "extraction",
            ),
            rewrite_agent_definition_ref=_ref(
                "agent-definition",
                "rewrite",
            ),
            allowed_model_profile_refs=profile_refs,
            intent_budget_reservation_ref=_ref(
                "work-model-reservation",
                "extraction",
            ),
            rewrite_budget_reservation_ref=_ref(
                "work-model-reservation",
                "rewrite",
            ),
        ),
    )
    planner = GatewayRequirementPlannerAgent(
        gateway=gateway,
        private_store=private_store,
        config=RequirementPlannerConfig(
            prompt=prompts["planning"],
            agent_definition_ref=_ref(
                "agent-definition",
                "planning",
            ),
            allowed_model_profile_refs=profile_refs,
            budget_reservation_ref=_ref(
                "work-model-reservation",
                "planning",
            ),
        ),
    )
    core_runner = CoreVerticalRunner(
        workspace=tmp_path / "core-workspace",
        private_store=private_store,
        semantic_agent=semantic_agent,
    )
    r4_agent, r4_provider = _r4_agent(
        tmp_path / "r4",
        private_store,
    )
    store = FactoryControlStore(tmp_path / "factory.sqlite3")
    compiler = DatasetBuildPlanCompiler(_registry())
    reviews = PlanReviewService(store, compiler=compiler)
    task_materials = FactoryTaskAuthoringMaterialStore(private_store)
    task_bridge = FactoryTaskAuthoringBridge(
        core_runner=core_runner,
        private_store=private_store,
        agent=r4_agent,
    )
    job_store = JobStore(tmp_path / "factory-job.sqlite3")
    job_bridge = FactoryJobStoreBridge(
        job_store=job_store,
        template=_job_template(),
    )
    runtime = FactoryDatasetRuntime(
        store=store,
        planner=planner,
        compiler=compiler,
        plan_reviews=reviews,
        requested_by=USER,
        core_runner=core_runner,
        core_materials=FactoryCoreMaterialStore(store),
        task_authoring_bridge=task_bridge,
        task_authoring_materials=task_materials,
        job_store_bridge=job_bridge,
    )
    request = _request(manifest_sha256)
    waiting = await runtime.advance(
        request=request,
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
            idempotency_key="approve-item-runtime",
        ),
        audit=_audit(),
    )
    reviews.resume(
        review_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=1,
            resumed_by=USER,
            idempotency_key="resume-item-runtime",
        ),
        audit=_audit(),
    )
    core_input = FactoryDatasetCoreInput(
        manifest_path=manifest,
        raw_root=manifest.parent,
        expected_manifest_sha256=manifest_sha256,
    )

    first = await runtime.advance(
        request=request,
        policy=_policy(),
        requirement=_requirement(),
        core_input=core_input,
        audit=_audit(),
    )
    core_calls = core_provider.calls
    r4_calls = r4_provider.calls
    replay = await runtime.advance(
        request=request,
        policy=_policy(),
        requirement=_requirement(),
        core_input=core_input,
        audit=_audit(),
    )

    assert replay == first
    assert core_provider.calls == core_calls == 3
    assert r4_provider.calls == r4_calls == 4
    binding = store.list_item_bindings(request.dataset_run_id)[0]
    core_head = store.get_item_stage_head(
        binding.item_id,
        FactoryItemStageV2.CORE_SELECTION,
    )
    task_head = store.get_item_stage_head(
        binding.item_id,
        FactoryItemStageV2.TASK_AUTHORING,
    )
    assert task_head.dependency_result_refs == (core_head.result_ref,)
    material_ref = store.get_item_stage_material_ref(task_head.to_ref())
    restored = task_materials.get(material_ref)
    assert task_head.result_ref.object_type == ("r4-task-contract-set")
    assert task_head.result_ref.object_id == restored.task_contract_set.contract_set_id
    assert binding.item_id == dataset_item_id_v2(
        request.dataset_run_id,
        restored.source_trace_ref,
    )
    job_authority = job_bridge.prepare(
        dataset_run_id=request.dataset_run_id,
        source_trace_refs=(restored.source_trace_ref,),
        audit=_audit(),
    )
    assert job_authority.graph.item_ids == (binding.item_id,)
    assert job_store.get_job(request.dataset_run_id).status is JobStatus.RUNNING
    assert job_store.get_item(binding.item_id).status is ItemStatus.RUNNING

    ledger_ref = FacadeObjectRef(
        object_type="world-ledger",
        object_id="world-ledger://dataset-item-runtime",
        object_version="v2",
        object_sha256=hashlib.sha256(b"dataset-item-runtime-ledger").hexdigest(),
    )
    preparation = await FactoryAttachmentPreparationBuilder(
        FactoryAttachmentPreparationConfig(
            retrieval_policy=_retrieval_policy(),
            routing_policy=_routing_policy(),
            world_ledger_ref=ledger_ref,
        )
    ).build(
        task_authoring=restored,
        routing_facade=_RoutingFacade(),
        execution_facade=_SnapshotFacade(),
        audit=_audit(),
    )

    AttachmentR5PreparationValidator().validate_current(preparation)
    assert preparation.execution_result.execution_plan is not None
    assert len(preparation.execution_result.execution_plan.groups) == 1
    preparation_materials = FactoryAttachmentPreparationMaterialStore(private_store)
    preparation_ref = preparation_materials.put(preparation)
    assert preparation_materials.get(preparation_ref) == preparation
    with pytest.raises(
        FactoryAttachmentPreparationMaterialError,
        match="reference type",
    ):
        preparation_materials.get(preparation_ref.model_copy(update={"object_type": "wrong-material"}))

    attachment_planner, attachment_provider = _attachment_planning_agent(
        tmp_path,
        private_store,
    )
    routing_facade = _RoutingFacade()
    snapshot_facade = _SnapshotFacade()
    attachment_runtime = FactoryAttachmentItemRuntime(
        store=store,
        plan_reviews=reviews,
        planner=attachment_planner,
        preparation_builder=(
            FactoryAttachmentPreparationBuilder(
                FactoryAttachmentPreparationConfig(
                    retrieval_policy=_retrieval_policy(),
                    routing_policy=_routing_policy(),
                    world_ledger_ref=ledger_ref,
                )
            )
        ),
        preparation_materials=preparation_materials,
        routing_facade=routing_facade,
        execution_facade=snapshot_facade,
        requested_by=USER,
        runner=SupervisedAttachmentR5Runner(
            supervisor=ExecutionSupervisor(
                store,
                _attachment_registry(),
                workspace_manager=AgentWorkspaceManager(tmp_path / "attachment-workspaces"),
            ),
            review_service=reviews,
        ),
    )
    composed_runtime = FactoryDatasetRuntime(
        store=store,
        planner=planner,
        compiler=compiler,
        plan_reviews=reviews,
        requested_by=USER,
        core_runner=core_runner,
        core_materials=FactoryCoreMaterialStore(store),
        task_authoring_bridge=task_bridge,
        task_authoring_materials=task_materials,
        attachment_runtime=attachment_runtime,
        job_store_bridge=job_bridge,
    )
    composed_waiting = await composed_runtime.advance(
        request=request,
        policy=_policy(),
        requirement=_requirement(),
        core_input=core_input,
        audit=_audit(),
    )
    waiting_attachment = await attachment_runtime.advance(
        binding=binding,
        task_authoring=restored,
        policy=_policy(),
        audit=_audit(),
    )
    replay_attachment = await attachment_runtime.advance(
        binding=binding,
        task_authoring=restored,
        policy=_policy(),
        audit=_audit(),
    )

    assert replay_attachment == waiting_attachment
    assert composed_waiting.pending_review_refs == (waiting_attachment.review_ref,)
    assert composed_waiting.next_action.value == "REVIEW_PLAN"
    assert waiting_attachment.state is FactoryAttachmentItemState.WAITING_REVIEW
    assert attachment_provider.calls == 1
    assert routing_facade.calls == 1
    assert snapshot_facade.calls == 1
    assert snapshot_facade.execution_calls == 0
    assert (
        store.get_domain_plan_material_ref(
            run_id=store.get_item_run(binding).run_id,
            plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
        )
        == preparation_ref
    )

    attachment_review_id = waiting_attachment.review_ref.object_id
    reviews.decide(
        attachment_review_id,
        PlanReviewDecisionSubmissionV1(
            expected_plan_version=1,
            decision=PlanDecisionKindV2.APPROVE,
            decided_by=USER,
            reason_code="ATTACHMENT_PLAN_APPROVED",
            idempotency_key="approve-item-attachment",
        ),
        audit=_audit(),
    )
    reviews.resume(
        attachment_review_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=1,
            resumed_by=USER,
            idempotency_key="resume-item-attachment",
        ),
        audit=_audit(),
    )
    composed_resumed = await composed_runtime.advance(
        request=request,
        policy=_policy(),
        requirement=_requirement(),
        core_input=core_input,
        audit=_audit(),
    )
    executed_attachment = await attachment_runtime.advance(
        binding=binding,
        task_authoring=restored,
        policy=_policy(),
        audit=_audit(),
    )

    assert executed_attachment.state is FactoryAttachmentItemState.EXECUTED
    assert composed_resumed.pending_review_refs == ()
    assert composed_resumed.next_action.value == "ADVANCE"
    attachment_head = store.get_item_stage_head(
        binding.item_id,
        FactoryItemStageV2.ATTACHMENT,
    )
    assert executed_attachment.stage_head_ref == (attachment_head.to_ref())
    assert attachment_head.dependency_result_refs == (task_head.result_ref,)
    assert (
        store.get_domain_result(
            store.get_item_run(binding).run_id,
            PlanKindV2.ATTACHMENT_GENERATION,
        ).result_ref
        == attachment_head.result_ref
    )
    assert attachment_provider.calls == 1
    assert routing_facade.calls == 1
    assert snapshot_facade.calls == 1
    assert snapshot_facade.execution_calls == 1

    executed_replay = await attachment_runtime.advance(
        binding=binding,
        task_authoring=restored,
        policy=_policy(),
        audit=_audit(),
    )
    assert executed_replay == executed_attachment
    assert attachment_provider.calls == 1
    assert routing_facade.calls == 1
    assert snapshot_facade.calls == 1
    assert snapshot_facade.execution_calls == 1
