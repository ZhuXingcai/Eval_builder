from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel
from test_support.team_runtime_fixtures import (
    audit,
    capability_runtime,
    ref,
    team_fixture,
)

from eval_factory.agent_system.dataset_graph_handlers import (
    FactoryDatasetGraphHandlers,
)
from eval_factory.agent_system.graph import FactoryGraphSignalV2
from eval_factory.agent_system.planner_assessment import (
    FactoryPlannerAssessmentCompiler,
    FactoryPlannerAssessmentFacts,
)
from eval_factory.contracts.agent_system_v2 import (
    CompiledDatasetBuildPlanV2,
    DatasetBuildPlanTaskV2,
    DatasetBuildPlanV2,
    FactoryRunStatusV2,
    FactoryRunV2,
    PlanKindV2,
    PlannerAssessmentActionV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.harness import (
    ArtifactEnvelopeV1,
    ArtifactModalityV1,
)
from eval_factory.harness.contracts import sorted_refs
from eval_factory.packs import (
    build_generic_agent_evaluation_blueprint,
)
from eval_factory.packs.generic_agent_trace.capability_contracts import (
    AttachmentQualityCapabilityRequestV1,
    AttachmentReconstructionCapabilityRequestV1,
    BatchQualityCapabilityRequestV1,
    CriteriaRubricCapabilityRequestV1,
    DeliveryCapabilityRequestV1,
    GradingDesignCapabilityRequestV1,
    PlanReviewCapabilityRequestV1,
    RequirementPlanningCapabilityRequestV1,
    TaskAuthoringCapabilityRequestV1,
    TraceIngestionCapabilityRequestV1,
)
from eval_factory.packs.generic_agent_trace.factory_workflow import (
    GenericAgentFactoryWorkflow,
    GenericAgentTeamTaskSpec,
    materialize_generic_agent_team,
)
from eval_factory.team import (
    TeamCapabilityRunner,
    TeamSnapshot,
    TeamStore,
    TeamTaskWork,
)


class _MechanismRequests:
    def build(
        self,
        *,
        snapshot: TeamSnapshot,
        work: TeamTaskWork,
        spec: GenericAgentTeamTaskSpec,
    ) -> BaseModel:
        del snapshot, work
        capability_id = spec.capability_id
        if capability_id == "capability.requirement-planning":
            return RequirementPlanningCapabilityRequestV1.create(
                plan_ref=ref(
                    "dataset-build-plan",
                    "mechanism",
                    version="v2",
                ),
                policy_ref=ref(
                    "factory-run-policy",
                    "mechanism",
                    version="v2",
                ),
                audit=audit(),
            )
        if capability_id == "capability.trace-ingestion":
            return TraceIngestionCapabilityRequestV1.create(
                trace_source_ref=ref(
                    "trace-source",
                    "mechanism",
                    version="v2",
                ),
                job_id="mechanism-job",
                attempt=1,
                audit=audit(),
            )
        if capability_id == "capability.task-authoring":
            return TaskAuthoringCapabilityRequestV1.create(
                decision_ref=ref(
                    "trace-candidate-decision",
                    "mechanism",
                    version="v2",
                ),
                extracted_prompt_ref=ref(
                    "extracted-user-prompt",
                    "mechanism",
                    version="v2",
                ),
                intent_ref=ref(
                    "inferred-user-intent",
                    "mechanism",
                    version="v2",
                ),
                rewrite_ref=ref(
                    "task-rewrite-candidate",
                    "mechanism",
                    version="v2",
                ),
                requirement_ref=ref(
                    "evaluation-requirement-spec",
                    "mechanism",
                    version="v2",
                ),
                audit=audit(),
            )
        if capability_id == ("capability.attachment-reconstruction"):
            return AttachmentReconstructionCapabilityRequestV1.create(
                run_id="mechanism-run",
                plan_ref=ref(
                    "attachment-generation-plan",
                    "mechanism",
                    version="v2",
                ),
                compiled_plan_ref=ref(
                    "compiled-attachment-generation-plan",
                    "mechanism",
                    version="v2",
                ),
                preparation_ref=ref(
                    "attachment-r5-preparation",
                    "mechanism",
                    version="v2",
                ),
                prior_batch_ref=None,
                lease_duration_seconds=60,
                audit=audit(),
            )
        if capability_id == "capability.criteria-rubric":
            return CriteriaRubricCapabilityRequestV1.create(
                run_id="mechanism-run",
                plan_ref=ref(
                    "criteria-rubric-plan",
                    "mechanism",
                    version="v2",
                ),
                compiled_plan_ref=ref(
                    "compiled-criteria-rubric-plan",
                    "mechanism",
                    version="v2",
                ),
                task_draft_ref=ref(
                    "task-draft",
                    "mechanism",
                    version="v2",
                ),
                attachment_quality_ref=ref(
                    "attachment-quality-assessment",
                    "mechanism",
                    version="v2",
                ),
                solvability_ref=ref(
                    "solvability-assessment",
                    "mechanism",
                    version="v2",
                ),
                binding_definitions_ref=ref(
                    "evaluator-binding-definitions",
                    "mechanism",
                    version="v2",
                ),
                tool_catalog_ref=ref(
                    "tool-capability-catalog",
                    "mechanism",
                    version="v2",
                ),
                lease_duration_seconds=60,
                audit=audit(),
            )
        if capability_id == "capability.grading-design":
            return GradingDesignCapabilityRequestV1.create(
                run_id="mechanism-run",
                plan_ref=ref(
                    "grading-design-plan",
                    "mechanism",
                    version="v2",
                ),
                compiled_plan_ref=ref(
                    "compiled-grading-design-plan",
                    "mechanism",
                    version="v2",
                ),
                criteria_result_ref=ref(
                    "criteria-rubric-result",
                    "mechanism",
                    version="v2",
                ),
                criteria_route_ref=ref(
                    "model-route-decision",
                    "mechanism",
                    version="v2",
                ),
                rubric_set_ref=ref(
                    "rubric-set",
                    "mechanism",
                    version="v2",
                ),
                evaluator_spec_ref=ref(
                    "evaluator-spec",
                    "mechanism",
                    version="v2",
                ),
                reference_policy_ref=ref(
                    "reference-policy",
                    "mechanism",
                    version="v2",
                ),
                tool_policy_ref=ref(
                    "tool-policy",
                    "mechanism",
                    version="v2",
                ),
                model_authorizations_ref=ref(
                    "model-domain-authorizations",
                    "mechanism",
                    version="v2",
                ),
                evaluated_at=datetime(2026, 8, 20, tzinfo=UTC),
                lease_duration_seconds=60,
                audit=audit(),
            )
        if capability_id == "capability.quality-review":
            return AttachmentQualityCapabilityRequestV1.create(
                subgraph_result_ref=ref(
                    "attachment-subgraph-result",
                    "mechanism",
                    version="v2",
                ),
                group_result_refs=(
                    ref(
                        "attachment-group-result",
                        "mechanism",
                        version="v2",
                    ),
                ),
                work_envelope_refs=(
                    ref(
                        "agent-result-envelope",
                        "mechanism",
                        version="v2",
                    ),
                ),
                source_validation_ref=ref(
                    "deterministic-item-validation-result",
                    "mechanism",
                    version="v2",
                ),
                item_quality_ref=ref(
                    "item-quality-compilation-result",
                    "mechanism",
                    version="v2",
                ),
                audit=audit(),
            )
        if capability_id == "capability.batch-quality":
            return BatchQualityCapabilityRequestV1.create(
                dataset_run_id="mechanism-run",
                core_vertical_result_ref=ref(
                    "core-vertical-result",
                    "mechanism",
                    version="v2",
                ),
                sources_ref=ref(
                    "batch-quality-sources",
                    "mechanism",
                    version="v2",
                ),
                rejected_binding_refs=(),
                blocked_binding_refs=(),
                resolved_job_work_graph_ref=ref(
                    "resolved-job-work-graph",
                    "mechanism",
                    version="v2",
                ),
                duplicate_policy_ref=ref(
                    "duplicate-detection-policy",
                    "mechanism",
                    version="v2",
                ),
                cross_item_policy_ref=ref(
                    "cross-item-safety-policy",
                    "mechanism",
                    version="v2",
                ),
                batch_policy_ref=ref(
                    "batch-quality-policy",
                    "mechanism",
                    version="v2",
                ),
                audit=audit(),
            )
        if capability_id == "capability.plan-review":
            return PlanReviewCapabilityRequestV1.create(
                run_id="mechanism-run",
                plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
                plan_ref=ref(
                    "attachment-generation-plan",
                    "mechanism",
                    version="v2",
                ),
                audit=audit(),
            )
        if capability_id == "capability.delivery":
            return DeliveryCapabilityRequestV1.create(
                dataset_run_id="mechanism-run",
                aggregate_ref=ref(
                    "factory-dataset-aggregate-result",
                    "mechanism",
                    version="v2",
                ),
                candidate_projection_refs=(
                    ref(
                        "release-projection-result",
                        "mechanism",
                        version="v2",
                    ),
                ),
                export_refs=(
                    ref(
                        "authorized-candidate-export",
                        "mechanism",
                        version="v2",
                    ),
                ),
                policy_ref=ref(
                    "factory-run-policy",
                    "mechanism",
                    version="v2",
                ),
                audit=audit(),
            )
        raise AssertionError(capability_id)


@pytest.mark.asyncio
async def test_fixed_pack_executes_all_ten_team_capabilities(
    tmp_path: Path,
) -> None:
    base = team_fixture()
    sessions = {
        role: ref("harness-session", f"mechanism-{role}")
        for role in (
            "control",
            "coordinator",
            "quality",
            "requirement",
            "task",
            "trace",
        )
    }
    authority = materialize_generic_agent_team(
        registration=base.registration,
        requirement_ref=ref(
            "evaluation-requirement-spec",
            "mechanism",
            version="v2",
        ),
        independent_session_refs=sessions,
        team_id="team-mechanism",
        team_incarnation_id="team-mechanism-incarnation-1",
        max_model_requests=100,
        max_model_tokens=1_000_000,
        max_cost_micro_usd=100_000_000,
        audit=audit(),
    )
    store = TeamStore(tmp_path / "team.sqlite3")
    store.create_team(
        team=authority.team,
        roster=authority.roster,
        graph=authority.graph,
        authority=authority.authority,
        idempotency_key="create-team",
    )
    definition = next(
        value
        for value in base.registration.capability_definitions
        if value.capability_id == "capability.requirement-planning"
    )
    requirement_envelope = ArtifactEnvelopeV1.create(
        artifact_id="mechanism-requirement-input",
        subject_ref=authority.team.goal_ref,
        schema_ref=definition.request_schema_ref,
        content_ref=authority.team.goal_ref,
        media_type="application/json",
        modality=ArtifactModalityV1.DOCUMENT,
        domain_tags=("generic-agent-trace",),
        semantic_role="evaluation-requirement",
        purpose="evaluation-data-production",
        classification="INTERNAL",
        lineage_refs=(),
        producer_capability_ref=definition.to_ref(),
        producer_task_ref=None,
        validation_refs=(),
        revision=1,
        predecessor_envelope_ref=None,
        audit=audit(),
    )
    store.seed_artifact_head(
        team_ref=authority.team.to_ref(),
        head_id="team-mechanism.head-evaluation-requirement",
        envelope=requirement_envelope,
        audit=audit(),
        idempotency_key="seed-requirement",
    )
    seeded = store.get_snapshot(authority.team.team_id)
    blueprint = build_generic_agent_evaluation_blueprint(
        registration=base.registration,
        requirement_ref=seeded.team.goal_ref,
        team_ref=seeded.team.to_ref(),
        roster_ref=seeded.roster.to_ref(),
        task_graph_ref=seeded.graph.to_ref(),
        authority_ref=seeded.authority.to_ref(),
        audit=audit(),
    )
    registry, runtime, providers = capability_runtime(base)
    workflow = GenericAgentFactoryWorkflow(
        blueprint=blueprint,
        store=store,
        runner=TeamCapabilityRunner(
            store=store,
            runtime=runtime,
            registry=registry,
        ),
        request_source=_MechanismRequests(),
    )
    assessment_task = DatasetBuildPlanTaskV2(
        task_key="team-workflow",
        stage="factory",
        task_kind="team-workflow",
        agent_role="coordinator",
        dependency_task_keys=(),
        input_object_types=("evaluation-requirement-spec",),
        output_object_types=("candidate-dataset-delivery",),
        required_capability_ids=("capability.delivery",),
        acceptance_check_refs=(
            ref(
                "acceptance-check",
                "mechanism-assessment",
                version="v2",
            ),
        ),
        plan_review_kind=None,
        max_attempts=3,
        max_model_requests=100,
        max_model_tokens=1_000_000,
        max_cost_micro_usd=100_000_000,
    )
    assessment_run_ref = ref(
        "factory-run",
        "mechanism-assessment",
        version="v2",
    )
    assessment_policy_ref = ref(
        "factory-run-policy",
        "mechanism-assessment",
        version="v2",
    )
    assessment_plan = DatasetBuildPlanV2.create(
        plan_id="plan.mechanism-assessment",
        run_ref=assessment_run_ref,
        plan_version=1,
        predecessor_plan_ref=None,
        goals=("Complete the fixed Team workflow.",),
        user_constraints=(),
        assumptions=(),
        unresolved_questions=(),
        stage_order=("factory",),
        tasks=(assessment_task,),
        required_review_kinds=(),
        total_model_requests=100,
        total_model_tokens=1_000_000,
        total_cost_micro_usd=100_000_000,
        audit=audit(),
    )
    assessment_compiled = CompiledDatasetBuildPlanV2.create(
        compiled_plan_id="compiled-plan.mechanism-assessment",
        source_plan_ref=assessment_plan.to_ref(),
        policy_ref=assessment_policy_ref,
        tasks=assessment_plan.tasks,
        topological_task_keys=(assessment_task.task_key,),
        audit=audit(),
    )
    assessment_run = FactoryRunV2.create(
        run_id="factory-run.mechanism-assessment",
        run_version=1,
        status=FactoryRunStatusV2.RUNNING,
        policy_ref=assessment_policy_ref,
        requirement_spec_ref=authority.team.goal_ref,
        current_plan_ref=assessment_plan.to_ref(),
        compiled_plan_ref=assessment_compiled.to_ref(),
        active_task_refs=(),
        result_refs=(),
        pending_review_ref=None,
        planner_assessment_ref=None,
        completion_ref=None,
        delivery_manifest_ref=None,
        transition_count=1,
        model_requests_used=0,
        model_tokens_used=0,
        cost_micro_usd_used=0,
        audit=audit(),
    )
    initial_checkpoint = store.create_checkpoint(
        authority.team.team_id,
        audit=audit(),
        idempotency_key="assessment-initial",
    )
    initial_work = store.list_task_work(
        authority.team.team_id,
    )
    initial_assessment = FactoryPlannerAssessmentCompiler().compile(
        FactoryPlannerAssessmentFacts(
            run=assessment_run,
            compiled_plan=assessment_compiled,
            validator_result_refs=(initial_checkpoint.to_ref(),),
            pending_task_refs=sorted_refs(
                value.task.to_ref() for value in initial_work if value.status.value == "PENDING"
            ),
            ready_task_refs=sorted_refs(
                value.task.to_ref() for value in initial_work if value.status.value == "READY"
            ),
        ),
        audit=audit(),
    )
    assert initial_assessment.proposed_action is PlannerAssessmentActionV2.CONTINUE

    execution_capabilities = frozenset(
        {
            "capability.requirement-planning",
            "capability.trace-ingestion",
            "capability.task-authoring",
            "capability.attachment-reconstruction",
            "capability.criteria-rubric",
            "capability.grading-design",
            "capability.quality-review",
            "capability.batch-quality",
        }
    )
    completed: list[ObjectRef] = []
    for _ in range(8):
        facts = await workflow.dispatch_ready(
            authority.team.team_id,
            audit=audit(),
            limit=1,
            capability_ids=execution_capabilities,
        )
        completed.extend(facts.completed_result_refs)

    assert sum(value.calls for value in providers.values()) == 8
    assert providers["capability.plan-review"].calls == 0
    assert providers["capability.delivery"].calls == 0

    for capability_id in (
        "capability.plan-review",
        "capability.delivery",
    ):
        facts = await workflow.dispatch_capability(
            authority.team.team_id,
            capability_id,
            audit=audit(),
        )
        completed.extend(facts.completed_result_refs)

    assert len(completed) == 10
    assert sum(value.calls for value in providers.values()) == 10
    assert all(value.calls == 1 for value in providers.values())
    assert not store.ready_tasks(authority.team.team_id)
    assert all(value.status.value == "COMPLETED" for value in store.list_task_work(authority.team.team_id))
    assert len(store.current_artifact_heads(authority.team.team_id)) == 11
    exhausted = await workflow.dispatch_ready(
        authority.team.team_id,
        audit=audit(),
        limit=1,
    )
    assert exhausted.attempted_task_refs == ()
    with pytest.raises(ValueError, match="between 1 and 10"):
        await workflow.dispatch_ready(
            authority.team.team_id,
            audit=audit(),
            limit=0,
        )
    with pytest.raises(ValueError, match="must be installed"):
        await workflow.dispatch_ready(
            authority.team.team_id,
            audit=audit(),
            capability_ids=frozenset({"capability.unknown"}),
        )
    replayed_control = await workflow.dispatch_capability(
        authority.team.team_id,
        "capability.delivery",
        audit=audit(),
    )
    assert replayed_control.attempted_task_refs == ()
    assert providers["capability.delivery"].calls == 1
    final_checkpoint = store.current_checkpoint(
        authority.team.team_id,
    )
    assert final_checkpoint is not None
    final_assessment = FactoryPlannerAssessmentCompiler().compile(
        FactoryPlannerAssessmentFacts(
            run=assessment_run,
            compiled_plan=assessment_compiled,
            validator_result_refs=(final_checkpoint.to_ref(),),
        ),
        audit=audit(),
    )
    assert final_assessment.proposed_action is PlannerAssessmentActionV2.FINISH
    assert (
        FactoryDatasetGraphHandlers.signal_from_assessment(
            initial_assessment,
        )
        is FactoryGraphSignalV2.CONTINUE
    )
    assert (
        FactoryDatasetGraphHandlers.signal_from_assessment(
            final_assessment,
        )
        is FactoryGraphSignalV2.FINISH
    )
