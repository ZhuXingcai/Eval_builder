from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.agent_system.core_vertical import CoreVerticalRunner
from eval_factory.agent_system.output import CoreOutputAssembler
from eval_factory.agent_system.private_store import FactoryPrivateObjectStore
from eval_factory.agent_system.specialists import (
    CoreSemanticInputV1,
    CoreSemanticOperationV1,
    GatewayIntentRewriteAgent,
    GatewayRequirementPlannerAgent,
    RequirementPlannerConfig,
    RequirementPlanningInputV1,
    SemanticAgentConfig,
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
    CompiledDatasetBuildPlanV2,
    DatasetBuildPlanTaskV2,
    DatasetBuildPlanV2,
    EvaluationRequirementSpecV2,
    FactoryCompletionOutcomeV2,
    FactoryRunPolicyV2,
    InferredUserIntentV2,
    IntentClaimV2,
    PlanKindV2,
    TaskRewriteCandidateV2,
    TraceCandidateDispositionV2,
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

ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT.parent / "raw_traj"
MANIFEST_SHA256 = "0d9df6c3935e00e5e15d486c9afe8cc2f581b63cf8f99ec42c1d3216cc74ee04"
HASH = "a" * 64
NOW = datetime(2026, 8, 6, tzinfo=UTC)

pytestmark = pytest.mark.skipif(
    not (RAW_ROOT / "manifest.csv").is_file(),
    reason="private 91-trace corpus is not installed",
)


def _ref(object_type: str, suffix: str = "example") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/v2",
        object_version="v2",
        object_sha256=HASH,
    )


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (ref.object_type, ref.object_id, ref.object_version, ref.object_sha256)


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="core-vertical-real-trace-test",
        governing_versions=(
            VersionBinding(
                component="graph-engineered-eval-factory",
                version="core-vertical-v1",
                sha256=HASH,
            ),
        ),
    )


def _profile(task_kind: str) -> ModelCapabilityProfileV2:
    return ModelCapabilityProfileV2.create(
        model_profile_id=f"model-profile://{task_kind}",
        provider_id="provider://offline-core",
        model_id=f"model://{task_kind}",
        model_version="v1",
        capabilities=(task_kind, "structured-output"),
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


def _prompt(task_kind: str) -> PromptTemplateV2:
    input_object_type = "evaluation-requirement-spec" if task_kind == "planning" else "extracted-user-prompt"
    return PromptTemplateV2.create(
        prompt_template_id=f"prompt-template://{task_kind}/core-v1",
        agent_role=f"{task_kind}-agent",
        task_kind=task_kind,
        system_template_ref=_ref("prompt-template-content", f"{task_kind}-system"),
        instruction_template_ref=_ref(
            "prompt-template-content",
            f"{task_kind}-instruction",
        ),
        required_input_object_types=(input_object_type,),
        output_schema_ref=_ref("json-schema", task_kind),
        allowed_tool_ids=(),
        required_model_capabilities=(task_kind, "structured-output"),
        injection_policy_ref=_ref("prompt-injection-policy"),
        template_version=1,
        audit=_audit(),
    )


class _CoreProvider:
    def __init__(self, private_store: FactoryPrivateObjectStore) -> None:
        self.private_store = private_store
        self.calls = 0

    async def invoke(
        self,
        request: GatewayInvocationRequestV2,
        *,
        model_profile: ModelCapabilityProfileV2,
    ) -> ProviderInvocationResult:
        self.calls += 1
        if "planning" in model_profile.capabilities:
            planning_input = self.private_store.get_model(
                request.prompt_rendering_ref,
                RequirementPlanningInputV1,
            )
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
                max_model_requests=2,
                max_model_tokens=4096,
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
                max_model_requests=2,
                max_model_tokens=4096,
                max_cost_micro_usd=100_000,
            )
            plan = DatasetBuildPlanV2.create(
                plan_id=f"dataset-build-plan://{planning_input.requirement_spec_ref.object_sha256[:32]}",
                run_ref=planning_input.run_ref,
                plan_version=1,
                predecessor_plan_ref=None,
                goals=planning_input.goals,
                user_constraints=planning_input.constraints,
                assumptions=planning_input.assumptions,
                unresolved_questions=planning_input.open_questions,
                stage_order=("core",),
                tasks=(extract, rewrite),
                required_review_kinds=(
                    PlanKindV2.GLOBAL_BUILD,
                    PlanKindV2.TASK_REWRITE,
                ),
                total_model_requests=4,
                total_model_tokens=8192,
                total_cost_micro_usd=200_000,
                audit=_audit(),
            )
            output_ref = self.private_store.put_model(
                object_type="dataset-build-plan",
                value=plan,
            )
            return self._success(output_ref)

        semantic_input = self.private_store.get_model(
            request.prompt_rendering_ref,
            CoreSemanticInputV1,
        )
        if semantic_input.operation is CoreSemanticOperationV1.INFER_INTENT:
            intent = InferredUserIntentV2.create(
                inferred_intent_id=(
                    f"inferred-user-intent://{semantic_input.extracted_prompt_ref.object_sha256[:32]}"
                ),
                extracted_prompt_ref=semantic_input.extracted_prompt_ref,
                claims=(
                    IntentClaimV2(
                        claim_id=(f"intent-claim://{semantic_input.extracted_prompt_ref.object_sha256[:32]}"),
                        summary="Complete the source-grounded user-requested task.",
                        evidence_refs=(semantic_input.extracted_prompt_ref,),
                        confidence_basis_points=8500,
                        uncertain=False,
                    ),
                ),
                unresolved_requirements=(),
                abstained=False,
                audit=_audit(),
            )
            output_ref = self.private_store.put_model(
                object_type="inferred-user-intent",
                value=intent,
            )
        else:
            assert semantic_input.inferred_intent_ref is not None
            assert semantic_input.rewrite_plan_ref is not None
            rewritten_prompt_ref = self.private_store.put_text(
                object_type="rewritten-prompt-content",
                text=(
                    "Evaluation task reconstructed from the source-bound user prompt:\n"
                    + semantic_input.prompt_text
                ),
            )
            candidate = TaskRewriteCandidateV2.create(
                candidate_id=(
                    f"task-rewrite-candidate://{semantic_input.extracted_prompt_ref.object_sha256[:32]}"
                ),
                extracted_prompt_ref=semantic_input.extracted_prompt_ref,
                inferred_intent_ref=semantic_input.inferred_intent_ref,
                rewrite_plan_ref=semantic_input.rewrite_plan_ref,
                rewritten_prompt_ref=rewritten_prompt_ref,
                evidence_refs=tuple(
                    sorted(
                        (
                            semantic_input.extracted_prompt_ref,
                            semantic_input.inferred_intent_ref,
                        ),
                        key=_ref_key,
                    )
                ),
                status="REVIEW_READY",
                audit=_audit(),
            )
            output_ref = self.private_store.put_model(
                object_type="task-rewrite-candidate",
                value=candidate,
            )
        return self._success(output_ref)

    def _success(self, output_ref: ObjectRef) -> ProviderInvocationResult:
        return ProviderInvocationResult(
            status=GatewayInvocationStatusV2.SUCCEEDED,
            response_body_ref=self.private_store.put_text(
                object_type="model-response-content",
                text=f"structured-output:{output_ref.object_id}",
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


def _gateway(
    tmp_path: Path,
) -> tuple[EmbeddedAIGateway, FactoryPrivateObjectStore, _CoreProvider, dict[str, object]]:
    private_store = FactoryPrivateObjectStore(tmp_path / "private")
    profiles = {name: _profile(name) for name in ("extraction", "planning", "rewrite")}
    prompts = {name: _prompt(name) for name in profiles}
    baselines = tuple(
        ModelQualityBaselineV2.create(
            baseline_id=f"model-quality-baseline://{name}",
            model_profile_ref=profile.to_ref(),
            task_kind=name,
            benchmark_ref=_ref("model-quality-benchmark", name),
            quality_basis_points=9000,
            minimum_sample_count=100,
            audit=_audit(),
        )
        for name, profile in profiles.items()
    )
    health = tuple(
        ModelHealthSnapshotV2.create(
            health_snapshot_id=f"model-health-snapshot://{name}",
            model_profile_ref=profile.to_ref(),
            observed_at=NOW,
            availability="HEALTHY",
            success_basis_points=9900,
            latency_milliseconds=10,
            audit=_audit(),
        )
        for name, profile in profiles.items()
    )
    prices = tuple(
        ModelPriceScheduleV2.create(
            price_schedule_id=f"model-price-schedule://{name}",
            model_profile_ref=profile.to_ref(),
            input_micro_usd_per_million_tokens=1,
            output_micro_usd_per_million_tokens=1,
            effective_from=NOW,
            audit=_audit(),
        )
        for name, profile in profiles.items()
    )
    profile_refs = tuple(sorted((profile.to_ref() for profile in profiles.values()), key=_ref_key))
    agent_refs = tuple(
        sorted(
            (
                _ref("agent-definition", "extraction"),
                _ref("agent-definition", "planning"),
                _ref("agent-definition", "rewrite"),
            ),
            key=_ref_key,
        )
    )
    policy = ModelRoutingPolicyV2.create(
        routing_policy_id="model-routing-policy://core-vertical",
        allowed_provider_ids=("provider://offline-core",),
        allowed_agent_definition_refs=agent_refs,
        allowed_model_profile_refs=profile_refs,
        quality_weight=100,
        health_weight=10,
        latency_weight=1,
        cost_weight=1,
        audit=_audit(),
    )
    catalog = ModelCatalog(
        profiles=tuple(profiles.values()),
        quality_baselines=baselines,
        health_snapshots=health,
        price_schedules=prices,
    )
    provider = _CoreProvider(private_store)
    gateway = EmbeddedAIGateway(
        router=ModelRouter(catalog, policy),
        catalog=catalog,
        prompts=PromptRegistry(tuple(prompts.values())),
        records=GatewayRecordStore(tmp_path / "gateway.sqlite3"),
        provider=provider,
        clock=lambda: NOW,
    )
    return (
        gateway,
        private_store,
        provider,
        {
            "profiles": profiles,
            "prompts": prompts,
        },
    )


@pytest.mark.asyncio
async def test_real_91_trace_core_vertical_has_unique_partition_and_exact_replay(
    tmp_path: Path,
) -> None:
    gateway, private_store, provider, values = _gateway(tmp_path)
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
            intent_agent_definition_ref=_ref("agent-definition", "extraction"),
            rewrite_agent_definition_ref=_ref("agent-definition", "rewrite"),
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
    requirement = EvaluationRequirementSpecV2.create(
        requirement_spec_id="evaluation-requirement-spec://real-91",
        run_id="factory-run://real-91",
        source_ref=_ref("evaluation-requirement-source", "real-91"),
        goals=("Select traces that can become source-grounded evaluation tasks.",),
        constraints=("Keep original user prompts immutable and source-bound.",),
        assumptions=("This 91-trace cohort validates closure, not quality.",),
        open_questions=("A future 300-trace cohort validates quality.",),
        requirement_version=1,
        audit=_audit(),
    )
    policy = FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://real-91",
        allowed_task_kinds=("task-rewrite", "trace-extraction"),
        max_transitions=256,
        max_plan_revisions=4,
        max_agent_attempts=2,
        max_model_requests=1_000,
        max_model_tokens=2_000_000,
        max_cost_micro_usd=25_000_000,
        audit=_audit(),
    )
    control_store = FactoryControlStore(tmp_path / "factory-control.sqlite3")
    run = control_store.create_run(
        policy=policy,
        requirement=requirement,
        idempotency_key="create-real-91-run",
    )
    planner = GatewayRequirementPlannerAgent(
        gateway=gateway,
        private_store=private_store,
        config=RequirementPlannerConfig(
            prompt=prompts["planning"],
            agent_definition_ref=_ref("agent-definition", "planning"),
            allowed_model_profile_refs=profile_refs,
            budget_reservation_ref=_ref("work-model-reservation", "planning"),
        ),
    )
    planned = await planner.propose(
        task_ref=_ref("agent-task", "planning"),
        run_ref=run.to_ref(),
        requirement=requirement,
        audit=_audit(),
    )
    planning_calls = provider.calls
    planned_replay = await planner.propose(
        task_ref=_ref("agent-task", "planning"),
        run_ref=run.to_ref(),
        requirement=requirement,
        audit=_audit(),
    )
    assert planned_replay == planned
    assert provider.calls == planning_calls == 1
    assert planned.plan.goals == requirement.goals
    assert planned.plan.user_constraints == requirement.constraints
    assert tuple(task.task_key for task in planned.plan.tasks) == ("extract", "rewrite")
    compiled = CompiledDatasetBuildPlanV2.create(
        compiled_plan_id="compiled-dataset-build-plan://real-91/v1",
        source_plan_ref=planned.plan.to_ref(),
        policy_ref=policy.to_ref(),
        tasks=planned.plan.tasks,
        topological_task_keys=("extract", "rewrite"),
        audit=_audit(),
    )
    control_store.commit_plan(
        run_id=run.run_id,
        expected_run_version=0,
        plan=planned.plan,
        compiled_plan=compiled,
        audit=_audit(),
        idempotency_key="commit-real-91-plan",
    )

    runner = CoreVerticalRunner(
        workspace=tmp_path / "vertical",
        private_store=private_store,
        semantic_agent=semantic_agent,
    )
    first = await runner.run(
        manifest_path=RAW_ROOT / "manifest.csv",
        raw_root=RAW_ROOT,
        expected_manifest_sha256=MANIFEST_SHA256,
        requirement=requirement,
        planning_route_ref=planned.route.to_ref(),
        audit=_audit(),
    )
    first_calls = provider.calls
    replay = await runner.run(
        manifest_path=RAW_ROOT / "manifest.csv",
        raw_root=RAW_ROOT,
        expected_manifest_sha256=MANIFEST_SHA256,
        requirement=requirement,
        planning_route_ref=planned.route.to_ref(),
        audit=_audit(),
    )

    assert replay == first
    assert first.result.source_count == 91
    assert (
        sum(
            len(values)
            for values in (
                first.result.candidate_decision_refs,
                first.result.non_candidate_decision_refs,
                first.result.blocked_decision_refs,
            )
        )
        == 91
    )
    assert {decision.source_trace_id for decision in first.decisions} == {
        f"source-trace://{row.instance_id}/{row.sid}" for row in _manifest_members()
    }
    assert all(decision.disposition is TraceCandidateDispositionV2.CANDIDATE for decision in first.decisions)
    assert len(first.extracted_prompts) == len(first.inferred_intents) == 91
    assert len(first.rewrite_candidates) == 91
    assert all(prompt.source_spans for prompt in first.extracted_prompts)
    assert all(
        candidate.extracted_prompt_ref != candidate.inferred_intent_ref
        for candidate in first.rewrite_candidates
    )
    assert len(first.result.route_decision_refs) == 183
    assert first_calls == 183
    assert provider.calls == first_calls
    output = CoreOutputAssembler(
        store=control_store,
        root=tmp_path / "core-output",
    )
    first_output = output.assemble(
        run_id=run.run_id,
        requirement=requirement,
        execution=first,
        audit=_audit(),
        idempotency_key="assemble-real-91-output",
    )
    replayed_output = output.assemble(
        run_id=run.run_id,
        requirement=requirement,
        execution=replay,
        audit=_audit(),
        idempotency_key="assemble-real-91-output",
    )
    assert first_output.view.delivery_manifest.item_count == 91
    assert first_output.view.completion.outcome is FactoryCompletionOutcomeV2.INCOMPLETE
    assert replayed_output.view == first_output.view
    assert replayed_output.reused is True
    assert provider.calls == first_calls


class _ManifestMember:
    def __init__(self, instance_id: str, sid: str) -> None:
        self.instance_id = instance_id
        self.sid = sid


def _manifest_members() -> tuple[_ManifestMember, ...]:
    import csv

    with (RAW_ROOT / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        return tuple(
            _ManifestMember(str(row["instance_id"]), str(row["sid"])) for row in csv.DictReader(stream)
        )
