from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from test_support.team_runtime_fixtures import (
    audit,
    capability_runtime,
    ref,
    team_fixture,
)

from eval_factory.agent_system.private_store import FactoryPrivateObjectStore
from eval_factory.contracts.agent_system_v2 import (
    DatasetBuildPlanTaskV2,
    DatasetBuildPlanV2,
    FactoryRunPolicyV2,
    PlanKindV2,
)
from eval_factory.packs.generic_agent_trace.capabilities import (
    GenericCapabilityAdapterError,
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
from eval_factory.packs.generic_agent_trace.material_resolver import (
    CompositeCapabilityMaterialResolver,
)
from eval_factory.packs.generic_agent_trace.request_factories import (
    CapabilityOwnerCollection,
    CapabilityOwnerMaterial,
    GenericAgentCapabilityPreparationService,
)
from eval_factory.packs.generic_agent_trace.request_preparation import (
    GenericAgentTaskRequestPreparer,
)
from eval_factory.packs.generic_agent_trace.request_store import (
    GenericAgentCapabilityRequestStore,
)
from eval_factory.packs.generic_agent_trace.task_graph_materializer import (
    GenericAgentTaskGraphMaterializer,
)


def _material(object_type: str, suffix: str) -> Any:
    return CapabilityOwnerMaterial(
        ref(object_type, suffix, version="v2"),
        object(),
    )


def _collection(object_type: str, suffix: str) -> Any:
    return CapabilityOwnerCollection(
        ref(object_type, suffix, version="v2"),
        (),
    )


def test_all_first_party_request_factories_build_provider_models() -> None:
    service = object.__new__(GenericAgentCapabilityPreparationService)
    requests: list[object] = []

    def capture(
        task_id: str,
        request: object,
        **kwargs: object,
    ) -> Any:
        del task_id, kwargs
        requests.append(request)
        return request

    service._commit = capture  # type: ignore[method-assign]

    service.prepare_requirement_planning(
        "requirement",
        plan=_material("dataset-build-plan", "plan"),
        policy=_material("factory-run-policy", "policy"),
        audit=audit(),
        idempotency_key="requirement",
    )
    service.prepare_trace_ingestion(
        "trace",
        trace_source=_material("trace-source", "source"),
        job_id="job-request-factory",
        attempt=1,
        audit=audit(),
        idempotency_key="trace",
    )
    service.prepare_task_authoring(
        "task",
        decision=_material("trace-candidate-decision", "decision"),
        extracted_prompt=_material("extracted-user-prompt", "prompt"),
        intent=_material("inferred-user-intent", "intent"),
        rewrite=_material("task-rewrite-candidate", "rewrite"),
        requirement=_material(
            "evaluation-requirement-spec",
            "requirement",
        ),
        audit=audit(),
        idempotency_key="task",
    )
    service.prepare_attachment_reconstruction(
        "attachment",
        run_id="factory-run://request-factory",
        plan=_material("attachment-generation-plan", "attachment-plan"),
        compiled_plan=_material(
            "compiled-attachment-generation-plan",
            "compiled-attachment",
        ),
        preparation=_material(
            "attachment-r5-preparation",
            "preparation",
        ),
        prior_batch=_material(
            "artifact-execution-batch",
            "prior-batch",
        ),
        lease_duration_seconds=600,
        audit=audit(),
        idempotency_key="attachment",
    )
    service.prepare_criteria_rubric(
        "criteria",
        run_id="factory-run://request-factory",
        plan=_material("criteria-rubric-plan", "criteria-plan"),
        compiled_plan=_material(
            "compiled-criteria-rubric-plan",
            "compiled-criteria",
        ),
        task_draft=_material("task-draft", "task-draft"),
        attachment_quality=_material(
            "attachment-quality-assessment",
            "attachment-quality",
        ),
        solvability=_material(
            "solvability-assessment",
            "solvability",
        ),
        binding_definitions=_collection(
            "evaluator-binding-definitions",
            "bindings",
        ),
        tool_catalog=_material(
            "tool-capability-catalog",
            "tool-catalog",
        ),
        lease_duration_seconds=600,
        audit=audit(),
        idempotency_key="criteria",
    )
    service.prepare_grading_design(
        "grading",
        run_id="factory-run://request-factory",
        plan=_material("grading-design-plan", "grading-plan"),
        compiled_plan=_material(
            "compiled-grading-design-plan",
            "compiled-grading",
        ),
        criteria_result=_material(
            "criteria-rubric-result",
            "criteria-result",
        ),
        criteria_route=_material(
            "model-route-decision",
            "criteria-route",
        ),
        rubric_set=_material("rubric-set", "rubric"),
        evaluator_spec=_material("evaluator-spec", "evaluator"),
        reference_policy=_material(
            "reference-policy",
            "reference-policy",
        ),
        tool_policy=_material("tool-policy", "tool-policy"),
        model_authorizations=_collection(
            "model-domain-authorizations",
            "authorizations",
        ),
        evaluated_at=datetime(2026, 8, 20, tzinfo=UTC),
        lease_duration_seconds=600,
        audit=audit(),
        idempotency_key="grading",
    )
    service.prepare_attachment_quality(
        "quality",
        subgraph_result=_material(
            "attachment-subgraph-result",
            "subgraph",
        ),
        group_results=(_material("attachment-group-result", "group"),),
        work_envelopes=(_material("agent-result-envelope", "work"),),
        source_validation=_material(
            "deterministic-item-validation-result",
            "validation",
        ),
        item_quality=_material(
            "item-quality-compilation-result",
            "quality",
        ),
        audit=audit(),
        idempotency_key="quality",
    )
    service.prepare_batch_quality(
        "batch",
        dataset_run_id="factory-run://request-factory",
        core_vertical_result_ref=ref(
            "core-vertical-result",
            "core",
            version="v2",
        ),
        sources=_collection("batch-quality-sources", "sources"),
        rejected_binding_refs=(),
        blocked_binding_refs=(),
        resolved_job_work_graph=_material(
            "resolved-job-work-graph",
            "graph",
        ),
        duplicate_policy=_material(
            "duplicate-detection-policy",
            "duplicates",
        ),
        cross_item_policy=_material(
            "cross-item-safety-policy",
            "cross-item",
        ),
        batch_policy=_material(
            "batch-quality-policy",
            "batch-policy",
        ),
        audit=audit(),
        idempotency_key="batch",
    )
    service.prepare_plan_review(
        "global-review",
        run_id="factory-run://request-factory",
        plan_kind=PlanKindV2.GLOBAL_BUILD,
        plan_ref=ref("dataset-build-plan", "plan", version="v2"),
        audit=audit(),
        idempotency_key="global-review",
    )
    service.prepare_delivery(
        "delivery",
        dataset_run_id="factory-run://request-factory",
        aggregate=_material(
            "factory-dataset-aggregate-result",
            "aggregate",
        ),
        candidates=(
            _material(
                "release-projection-result",
                "projection",
            ),
        ),
        exports=(
            _material(
                "authorized-candidate-export",
                "export",
            ),
        ),
        policy=_material("factory-run-policy", "delivery-policy"),
        audit=audit(),
        idempotency_key="delivery",
    )

    assert tuple(type(value) for value in requests) == (
        RequirementPlanningCapabilityRequestV1,
        TraceIngestionCapabilityRequestV1,
        TaskAuthoringCapabilityRequestV1,
        AttachmentReconstructionCapabilityRequestV1,
        CriteriaRubricCapabilityRequestV1,
        GradingDesignCapabilityRequestV1,
        AttachmentQualityCapabilityRequestV1,
        BatchQualityCapabilityRequestV1,
        PlanReviewCapabilityRequestV1,
        DeliveryCapabilityRequestV1,
    )


def test_owner_material_is_registered_before_request_binding(
    tmp_path: Path,
) -> None:
    base = team_fixture()
    materialized = GenericAgentTaskGraphMaterializer().materialize(
        registration=base.registration,
        requirement_ref=ref(
            "evaluation-requirement-spec",
            "request-factory",
            version="v2",
        ),
        source_refs=(ref("trace-source", "request-factory", version="v2"),),
        independent_session_refs={
            role: ref("harness-session", f"request-factory-{role}")
            for role in (
                "control",
                "coordinator",
                "quality",
                "requirement",
                "task",
                "trace",
            )
        },
        team_id="team-request-factory",
        team_incarnation_id="team-request-factory-incarnation-1",
        max_model_requests=100,
        max_model_tokens=1_000_000,
        max_cost_micro_usd=100_000_000,
        audit=audit(),
    )
    registry, _runtime, _providers = capability_runtime(base)
    request_store = GenericAgentCapabilityRequestStore(
        tmp_path / "requests.sqlite3",
        private_store=FactoryPrivateObjectStore(tmp_path / "private"),
    )
    preparer = GenericAgentTaskRequestPreparer(
        materialization=materialized,
        registry=registry,
        store=request_store,
    )
    resolver = CompositeCapabilityMaterialResolver()
    service = GenericAgentCapabilityPreparationService(
        preparer=preparer,
        materials=resolver,
    )
    run_ref = ref("factory-run", "request-factory", version="v2")
    policy = FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://request-factory",
        allowed_task_kinds=("trace-extraction",),
        max_transitions=64,
        max_plan_revisions=4,
        max_agent_attempts=2,
        max_model_requests=100,
        max_model_tokens=1_000_000,
        max_cost_micro_usd=100_000_000,
        audit=audit(),
    )
    task = DatasetBuildPlanTaskV2(
        task_key="trace",
        stage="core",
        task_kind="trace-extraction",
        agent_role="trace-agent",
        dependency_task_keys=(),
        input_object_types=("evaluation-requirement-spec",),
        output_object_types=("trace-envelope",),
        required_capability_ids=("capability.trace-ingestion",),
        acceptance_check_refs=(ref("acceptance-check", "request-factory", version="v2"),),
        plan_review_kind=PlanKindV2.GLOBAL_BUILD,
        max_attempts=2,
        max_model_requests=1,
        max_model_tokens=4096,
        max_cost_micro_usd=100_000,
    )
    plan = DatasetBuildPlanV2.create(
        plan_id="dataset-build-plan://request-factory",
        run_ref=run_ref,
        plan_version=1,
        predecessor_plan_ref=None,
        goals=("Build one governed trace candidate.",),
        user_constraints=(),
        assumptions=(),
        unresolved_questions=(),
        stage_order=("core",),
        tasks=(task,),
        required_review_kinds=(PlanKindV2.GLOBAL_BUILD,),
        total_model_requests=1,
        total_model_tokens=4096,
        total_cost_micro_usd=100_000,
        audit=audit(),
    )
    task_id = next(
        value.task_id
        for value in materialized.authority.graph.tasks
        if value.task_kind == "requirement-planning"
    )

    binding = service.prepare_requirement_planning(
        task_id,
        plan=CapabilityOwnerMaterial(plan.to_ref(), plan),
        policy=CapabilityOwnerMaterial(policy.to_ref(), policy),
        audit=audit(),
        idempotency_key="prepare-requirement",
    )
    request = request_store.load_request(
        binding,
        RequirementPlanningCapabilityRequestV1,
    )

    assert request.plan_ref == plan.to_ref()
    assert binding.source_authority_refs == request.audit.input_refs
    assert resolver.get(plan.to_ref(), DatasetBuildPlanV2) == plan
    assert resolver.get(policy.to_ref(), FactoryRunPolicyV2) == policy

    with pytest.raises(
        GenericCapabilityAdapterError,
        match="wrong owner type",
    ):
        service.prepare_requirement_planning(
            task_id,
            plan=CapabilityOwnerMaterial(plan.to_ref(), plan),
            policy=CapabilityOwnerMaterial(
                policy.to_ref(),
                cast(FactoryRunPolicyV2, plan),
            ),
            audit=audit(),
            idempotency_key="wrong-policy-owner",
        )
