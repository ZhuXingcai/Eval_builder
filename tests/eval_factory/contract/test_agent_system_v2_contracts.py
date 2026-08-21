from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ValidationError

import eval_factory.contracts.agent_system_v2 as contracts
from eval_factory.contracts.agent_system_v2 import (
    AgentCapabilityV2,
    AgentResultEnvelopeV2,
    AgentTaskOutcomeV2,
    AgentTaskStatusV2,
    AgentTaskV2,
    AgentWorkspaceReceiptV2,
    AgentWorkspaceStateV2,
    AttachmentGenerationPlanV2,
    AttachmentGroupResultV2,
    AttachmentMockWorkV2,
    AttachmentQualityAssessmentV2,
    AttachmentQualityOutcomeV2,
    AttachmentSubgraphOutcomeV2,
    AttachmentSubgraphResultV2,
    CoreVerticalResultV2,
    DatasetBuildPlanTaskV2,
    DatasetBuildPlanV2,
    ExtractedUserPromptV2,
    FactoryRunPolicyV2,
    FactoryRunStatusV2,
    FactoryRunV2,
    InferredUserIntentV2,
    IntentClaimV2,
    PlanKindV2,
    SolvabilityAssessmentV2,
    SolvabilityOutcomeV2,
    TaskRewriteCandidateV2,
    TaskRewritePlanV2,
    TraceCandidateDecisionV2,
    TraceCandidateDispositionV2,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    SourceSpanRef,
    VersionBinding,
)

HASH = "a" * 64


def _ref(object_type: str, suffix: str = "example") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/v2",
        object_version="v2",
        object_sha256=HASH,
    )


def _audit(actor: str = "agent-system-contract-test") -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 6, tzinfo=UTC),
        created_by=actor,
        governing_versions=(
            VersionBinding(
                component="graph-engineered-eval-factory",
                version="phase-a",
                sha256=HASH,
            ),
        ),
    )


def _sorted_refs(*refs: ObjectRef) -> tuple[ObjectRef, ...]:
    return tuple(
        sorted(
            refs,
            key=lambda value: (
                value.object_type,
                value.object_id,
                value.object_version,
                value.object_sha256,
            ),
        )
    )


def _task(
    task_key: str,
    *,
    dependencies: tuple[str, ...] = (),
) -> DatasetBuildPlanTaskV2:
    return DatasetBuildPlanTaskV2(
        task_key=task_key,
        stage="core",
        task_kind=f"{task_key}-kind",
        agent_role=f"{task_key}-agent",
        dependency_task_keys=dependencies,
        input_object_types=("requirement-spec",),
        output_object_types=(f"{task_key}-result",),
        required_capability_ids=(f"capability://{task_key}",),
        acceptance_check_refs=(_ref("acceptance-check", task_key),),
        plan_review_kind=PlanKindV2.GLOBAL_BUILD,
        max_attempts=2,
        max_model_requests=2,
        max_model_tokens=4096,
        max_cost_micro_usd=100_000,
    )


def test_factory_policy_is_strict_frozen_and_identity_derived() -> None:
    policy = FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://core-v1",
        allowed_task_kinds=("planning", "trace-extraction", "task-rewrite"),
        max_transitions=256,
        max_plan_revisions=8,
        max_agent_attempts=3,
        max_model_requests=1_000,
        max_model_tokens=2_000_000,
        max_cost_micro_usd=25_000_000,
        audit=_audit(),
    )

    assert policy.production_release_allowed is False
    assert policy.to_ref().object_sha256 == policy.object_sha256
    with pytest.raises(ValidationError):
        policy.max_transitions = 1  # type: ignore[misc]

    payload = policy.model_dump(mode="python")
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        FactoryRunPolicyV2.model_validate(payload)

    payload.pop("unexpected")
    payload["object_sha256"] = "b" * 64
    with pytest.raises(ValidationError, match="derived identity"):
        FactoryRunPolicyV2.model_validate(payload)


def test_plan_tasks_require_closed_canonical_dependencies() -> None:
    plan = DatasetBuildPlanV2.create(
        plan_id="dataset-build-plan://run-example",
        run_ref=_ref("factory-run"),
        plan_version=1,
        predecessor_plan_ref=None,
        goals=("screen real traces",),
        user_constraints=("preserve source evidence",),
        assumptions=("provider boundary is offline-faked",),
        unresolved_questions=("future 300-trace quality target",),
        stage_order=("core",),
        tasks=(_task("rewrite", dependencies=("extract",)), _task("extract")),
        required_review_kinds=(PlanKindV2.GLOBAL_BUILD,),
        total_model_requests=4,
        total_model_tokens=8192,
        total_cost_micro_usd=200_000,
        audit=_audit(),
    )
    assert tuple(task.task_key for task in plan.tasks) == ("extract", "rewrite")

    with pytest.raises(ValidationError, match="unknown dependency"):
        DatasetBuildPlanV2.create(
            plan_id="dataset-build-plan://invalid-dependency",
            run_ref=_ref("factory-run"),
            plan_version=1,
            predecessor_plan_ref=None,
            goals=("screen real traces",),
            user_constraints=(),
            assumptions=(),
            unresolved_questions=(),
            stage_order=("core",),
            tasks=(_task("rewrite", dependencies=("missing",)),),
            required_review_kinds=(),
            total_model_requests=2,
            total_model_tokens=4096,
            total_cost_micro_usd=100_000,
            audit=_audit(),
        )

    with pytest.raises(ValidationError, match="unique"):
        DatasetBuildPlanV2.create(
            plan_id="dataset-build-plan://duplicate-task",
            run_ref=_ref("factory-run"),
            plan_version=1,
            predecessor_plan_ref=None,
            goals=("screen real traces",),
            user_constraints=(),
            assumptions=(),
            unresolved_questions=(),
            stage_order=("core",),
            tasks=(_task("extract"), _task("extract")),
            required_review_kinds=(),
            total_model_requests=4,
            total_model_tokens=8192,
            total_cost_micro_usd=200_000,
            audit=_audit(),
        )


def test_run_state_contains_refs_and_bounded_control_values_only() -> None:
    policy = FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://core-v1",
        allowed_task_kinds=("planning",),
        max_transitions=32,
        max_plan_revisions=2,
        max_agent_attempts=2,
        max_model_requests=10,
        max_model_tokens=10_000,
        max_cost_micro_usd=1_000_000,
        audit=_audit(),
    )
    run = FactoryRunV2.create(
        run_id="factory-run://example",
        run_version=0,
        status=FactoryRunStatusV2.CREATED,
        policy_ref=policy.to_ref(),
        requirement_spec_ref=_ref("evaluation-requirement-spec"),
        current_plan_ref=None,
        compiled_plan_ref=None,
        active_task_refs=(),
        result_refs=(),
        pending_review_ref=None,
        planner_assessment_ref=None,
        completion_ref=None,
        delivery_manifest_ref=None,
        transition_count=0,
        model_requests_used=0,
        model_tokens_used=0,
        cost_micro_usd_used=0,
        audit=_audit(),
    )
    assert run.status is FactoryRunStatusV2.CREATED
    assert run.to_ref().object_type == "factory-run"
    assert run.audit.input_refs == (policy.to_ref(), _ref("evaluation-requirement-spec"))


def test_agent_task_outcome_cannot_be_fabricated_as_success() -> None:
    task = AgentTaskV2.create(
        agent_task_id="agent-task://extract/example",
        run_ref=_ref("factory-run"),
        compiled_plan_ref=_ref("compiled-dataset-build-plan"),
        plan_task_key="extract",
        task_kind="trace-extraction",
        agent_definition_ref=_ref("agent-definition"),
        input_refs=(_ref("trace-candidate-set"),),
        expected_output_object_types=("extracted-user-prompt",),
        required_capability_refs=(_ref("agent-capability"),),
        status=AgentTaskStatusV2.READY,
        attempt=0,
        max_attempts=2,
        audit=_audit(),
    )
    assert task.status is AgentTaskStatusV2.READY
    assert AgentTaskOutcomeV2.SUCCEEDED.value == "SUCCEEDED"

    payload = task.model_dump(mode="python")
    payload["status"] = "SUCCEEDED"
    with pytest.raises(ValidationError):
        AgentTaskV2.model_validate(payload)


def test_agent_task_accepts_compiled_criteria_rubric_plan() -> None:
    task = AgentTaskV2.create(
        agent_task_id="agent-task://criteria/example",
        run_ref=_ref("factory-run"),
        compiled_plan_ref=_ref("compiled-criteria-rubric-plan"),
        plan_task_key="criteria-rubric",
        task_kind="criteria-rubric",
        agent_definition_ref=_ref("agent-definition"),
        input_refs=(_ref("task-draft"),),
        expected_output_object_types=("criteria-rubric-result",),
        required_capability_refs=(_ref("agent-capability"),),
        status=AgentTaskStatusV2.READY,
        attempt=0,
        max_attempts=2,
        audit=_audit(),
    )

    assert task.compiled_plan_ref.object_type == ("compiled-criteria-rubric-plan")


def test_agent_result_delegates_jobstore_authority_by_immutable_ref_only() -> None:
    active_workspace = AgentWorkspaceReceiptV2.create(
        workspace_receipt_id="agent-workspace-receipt://active",
        run_ref=_ref("factory-run"),
        task_ref=_ref("agent-task"),
        agent_definition_ref=_ref("agent-definition"),
        attempt=1,
        fencing_token=1,
        namespace_sha256=HASH,
        state=AgentWorkspaceStateV2.ACTIVE,
        predecessor_workspace_receipt_ref=None,
        inventory_sha256=None,
        reason_code=None,
        audit=_audit(),
    )
    committed_workspace = AgentWorkspaceReceiptV2.create(
        workspace_receipt_id="agent-workspace-receipt://committed",
        run_ref=active_workspace.run_ref,
        task_ref=active_workspace.task_ref,
        agent_definition_ref=active_workspace.agent_definition_ref,
        attempt=active_workspace.attempt,
        fencing_token=active_workspace.fencing_token,
        namespace_sha256=active_workspace.namespace_sha256,
        state=AgentWorkspaceStateV2.COMMITTED,
        predecessor_workspace_receipt_ref=active_workspace.to_ref(),
        inventory_sha256=HASH,
        reason_code=None,
        audit=_audit(),
    )
    result = AgentResultEnvelopeV2.create(
        result_id="agent-result-envelope://example",
        task_ref=_ref("agent-task"),
        agent_definition_ref=_ref("agent-definition"),
        attempt=1,
        lease_id="agent-lease://example",
        fencing_token=1,
        workspace_receipt_ref=committed_workspace.to_ref(),
        outcome=AgentTaskOutcomeV2.SUCCEEDED,
        output_refs=(_ref("trace-candidate-set"),),
        delegated_dataset_job_result_refs=(_ref("stage-result"),),
        gateway_receipt_refs=(),
        validator_result_refs=(_ref("validator-result"),),
        failure_code=None,
        safe_metrics=(),
        audit=_audit(),
    )
    assert result.delegated_dataset_job_result_refs == (_ref("stage-result"),)

    with pytest.raises(ValidationError, match="unsupported result type"):
        AgentResultEnvelopeV2.create(
            result_id="agent-result-envelope://invalid-delegation",
            task_ref=_ref("agent-task"),
            agent_definition_ref=_ref("agent-definition"),
            attempt=1,
            lease_id="agent-lease://invalid",
            fencing_token=1,
            workspace_receipt_ref=committed_workspace.to_ref(),
            outcome=AgentTaskOutcomeV2.SUCCEEDED,
            output_refs=(_ref("trace-candidate-set"),),
            delegated_dataset_job_result_refs=(_ref("job-store-row"),),
            gateway_receipt_refs=(),
            validator_result_refs=(_ref("validator-result"),),
            failure_code=None,
            safe_metrics=(),
            audit=_audit(),
        )

    forbidden = {
        "workspace_path",
        "root_path",
        "physical_path",
    }
    assert forbidden.isdisjoint(AgentWorkspaceReceiptV2.model_fields)
    with pytest.raises(
        ValidationError,
        match="committed workspace",
    ):
        AgentWorkspaceReceiptV2.create(
            workspace_receipt_id=("agent-workspace-receipt://bad-commit"),
            run_ref=_ref("factory-run"),
            task_ref=_ref("agent-task"),
            agent_definition_ref=_ref("agent-definition"),
            attempt=1,
            fencing_token=1,
            namespace_sha256=HASH,
            state=AgentWorkspaceStateV2.COMMITTED,
            predecessor_workspace_receipt_ref=(active_workspace.to_ref()),
            inventory_sha256=None,
            reason_code=None,
            audit=_audit(),
        )


def test_user_prompt_is_source_bound_and_intent_is_evidence_bound() -> None:
    prompt = ExtractedUserPromptV2.create(
        extracted_prompt_id="extracted-user-prompt://example",
        trace_ref=_ref("trace-ir"),
        interaction_segment_ref=_ref("interaction-segment"),
        content_ref=_ref("user-prompt-content"),
        source_spans=(
            SourceSpanRef(
                span_id="source-span://example",
                source_trace_id="source-trace://example",
                raw_sha256=HASH,
            ),
        ),
        context_segment_refs=(_ref("interaction-segment", "context"),),
        audit=_audit(),
    )
    intent = InferredUserIntentV2.create(
        inferred_intent_id="inferred-user-intent://example",
        extracted_prompt_ref=prompt.to_ref(),
        claims=(
            IntentClaimV2(
                claim_id="intent-claim://goal",
                summary="Produce a source-grounded task rewrite.",
                evidence_refs=(_ref("intent-evidence"),),
                confidence_basis_points=9000,
                uncertain=False,
            ),
        ),
        unresolved_requirements=("attachment criteria are deferred",),
        abstained=False,
        audit=_audit(),
    )
    rewrite = TaskRewritePlanV2.create(
        rewrite_plan_id="task-rewrite-plan://example",
        extracted_prompt_ref=prompt.to_ref(),
        inferred_intent_ref=intent.to_ref(),
        rewrite_policy_ref=_ref("task-rewrite-policy"),
        target_capabilities=("instruction-following",),
        fidelity_constraints=("do not add unsupported facts",),
        forbidden_transformations=("replace original prompt with inferred intent",),
        acceptance_check_refs=(_ref("acceptance-check", "rewrite"),),
        audit=_audit(),
    )

    assert rewrite.extracted_prompt_ref == prompt.to_ref()
    assert rewrite.inferred_intent_ref == intent.to_ref()
    assert not hasattr(prompt, "prompt_text")

    with pytest.raises(ValidationError, match="non-abstained"):
        InferredUserIntentV2.create(
            inferred_intent_id="inferred-user-intent://invalid",
            extracted_prompt_ref=prompt.to_ref(),
            claims=(),
            unresolved_requirements=(),
            abstained=False,
            audit=_audit(),
        )


def test_core_vertical_contracts_bind_disjoint_partitions_and_exact_artifacts() -> None:
    candidate_decision = TraceCandidateDecisionV2.create(
        decision_id="trace-candidate-decision://candidate",
        source_trace_id="source-trace://candidate",
        source_ref=_ref("trace-source", "candidate"),
        disposition=TraceCandidateDispositionV2.CANDIDATE,
        reason_codes=(),
        cleaned_trace_ref=_ref("trace-ir", "candidate"),
        audit=_audit(),
    )
    prompt_ref = _ref("extracted-user-prompt", "candidate")
    intent_ref = _ref("inferred-user-intent", "candidate")
    rewrite_plan_ref = _ref("task-rewrite-plan", "candidate")
    rewrite = TaskRewriteCandidateV2.create(
        candidate_id="task-rewrite-candidate://candidate",
        extracted_prompt_ref=prompt_ref,
        inferred_intent_ref=intent_ref,
        rewrite_plan_ref=rewrite_plan_ref,
        rewritten_prompt_ref=_ref("rewritten-prompt-content", "candidate"),
        evidence_refs=_sorted_refs(prompt_ref, intent_ref),
        audit=_audit(),
    )
    result = CoreVerticalResultV2.create(
        result_id="core-vertical-result://candidate",
        manifest_ref=_ref("trace-manifest"),
        requirement_spec_ref=_ref("evaluation-requirement-spec"),
        candidate_decision_refs=(candidate_decision.to_ref(),),
        non_candidate_decision_refs=(),
        blocked_decision_refs=(),
        extracted_prompt_refs=(prompt_ref,),
        inferred_intent_refs=(intent_ref,),
        rewrite_candidate_refs=(rewrite.to_ref(),),
        route_decision_refs=_sorted_refs(
            _ref("model-route-decision", "extraction"),
            _ref("model-route-decision", "planning"),
            _ref("model-route-decision", "rewrite"),
        ),
        source_count=1,
        audit=_audit(),
    )

    assert result.source_count == 1
    assert rewrite.extracted_prompt_ref != rewrite.inferred_intent_ref

    with pytest.raises(ValidationError, match="exact coverage"):
        CoreVerticalResultV2.create(
            result_id="core-vertical-result://missing-artifacts",
            manifest_ref=_ref("trace-manifest"),
            requirement_spec_ref=_ref("evaluation-requirement-spec"),
            candidate_decision_refs=(candidate_decision.to_ref(),),
            non_candidate_decision_refs=(),
            blocked_decision_refs=(),
            extracted_prompt_refs=(),
            inferred_intent_refs=(),
            rewrite_candidate_refs=(),
            route_decision_refs=result.route_decision_refs,
            source_count=1,
            audit=_audit(),
        )


def test_trace_candidate_decision_rejects_partition_shape_drift() -> None:
    with pytest.raises(ValidationError, match="candidate trace"):
        TraceCandidateDecisionV2.create(
            decision_id="trace-candidate-decision://missing-cleaned-ref",
            source_trace_id="source-trace://missing-cleaned-ref",
            source_ref=_ref("trace-source", "missing-cleaned-ref"),
            disposition=TraceCandidateDispositionV2.CANDIDATE,
            reason_codes=("UNEXPECTED_REASON",),
            cleaned_trace_ref=None,
            audit=_audit(),
        )

    with pytest.raises(ValidationError, match="non-candidate trace"):
        TraceCandidateDecisionV2.create(
            decision_id="trace-candidate-decision://missing-reason",
            source_trace_id="source-trace://missing-reason",
            source_ref=_ref("trace-source", "missing-reason"),
            disposition=TraceCandidateDispositionV2.NON_CANDIDATE,
            reason_codes=(),
            cleaned_trace_ref=None,
            audit=_audit(),
        )


def test_public_contract_models_exclude_sensitive_content_fields() -> None:
    forbidden = {
        "credential",
        "model_output_body",
        "prompt_body",
        "rag_text",
        "raw_trace",
        "retrieved_text",
        "runtime_transcript",
    }
    model_types = [
        value
        for _, value in inspect.getmembers(contracts, inspect.isclass)
        if issubclass(value, BaseModel) and value.__module__ == contracts.__name__
    ]
    assert model_types
    for model_type in model_types:
        assert forbidden.isdisjoint(model_type.model_fields), model_type.__name__


def test_capability_sets_are_canonical_and_closed() -> None:
    capability = AgentCapabilityV2.create(
        capability_id="agent-capability://trace-extraction",
        task_kinds=("trace-extraction",),
        input_object_types=("trace-candidate-set",),
        output_object_types=("extracted-user-prompt",),
        model_capabilities=("structured-output", "text"),
        tool_ids=("trace-query",),
        data_purposes=("evaluation-dataset-construction",),
        data_classifications=("RESTRICTED_TRACE_DERIVED",),
        audit=_audit(),
    )
    assert capability.model_capabilities == ("structured-output", "text")

    with pytest.raises(ValidationError, match="unique"):
        AgentCapabilityV2.create(
            capability_id="agent-capability://invalid",
            task_kinds=("trace-extraction",),
            input_object_types=("trace-candidate-set",),
            output_object_types=("extracted-user-prompt",),
            model_capabilities=("structured-output",),
            tool_ids=("trace-query", "trace-query"),
            data_purposes=("evaluation-dataset-construction",),
            data_classifications=("RESTRICTED_TRACE_DERIVED",),
            audit=_audit(),
        )


def _attachment_work(
    work_key: str,
    *,
    artifact_ids: tuple[str, ...],
    dependencies: tuple[str, ...] = (),
) -> AttachmentMockWorkV2:
    return AttachmentMockWorkV2(
        work_key=work_key,
        artifact_group_ref=_ref("artifact-execution-group", work_key),
        artifact_ids=artifact_ids,
        agent_role="attachment-mock-agent",
        dependency_work_keys=dependencies,
        input_object_types=("attachment-planning-context",),
        output_object_types=("attachment-group-result",),
        required_capability_ids=("agent-capability://attachment-mock",),
        allowed_tool_ids=("attachment-execution",),
        data_purposes=("attachment-production",),
        data_classifications=("RESTRICTED_TRACE_DERIVED",),
        workspace_policy_ref=_ref("agent-workspace-policy"),
        acceptance_check_refs=(_ref("acceptance-check", work_key),),
        max_attempts=2,
        max_model_requests=1,
        max_model_tokens=2048,
        max_cost_micro_usd=50_000,
    )


def _attachment_plan(
    *,
    works: tuple[AttachmentMockWorkV2, ...] | None = None,
) -> AttachmentGenerationPlanV2:
    values = works or (
        _attachment_work("group-a", artifact_ids=("artifact-a",)),
        _attachment_work("group-b", artifact_ids=("artifact-b",)),
    )
    return AttachmentGenerationPlanV2.create(
        plan_id="attachment-generation-plan://example",
        run_ref=_ref("factory-run"),
        plan_version=1,
        predecessor_plan_ref=None,
        producer_task_view_ref=_ref("producer-task-view"),
        evidence_bundle_ref=ObjectRef(
            object_type="evidence-bundle",
            object_id="evidence-bundle://example/v1",
            object_version="v1",
            object_sha256=HASH,
        ),
        attachment_planning_context_ref=_ref("attachment-planning-context"),
        works=values,
        max_parallel_groups=2,
        quality_policy_ref=_ref("attachment-quality-policy"),
        solvability_policy_ref=_ref("solvability-policy"),
        total_model_requests=sum(value.max_model_requests for value in values),
        total_model_tokens=sum(value.max_model_tokens for value in values),
        total_cost_micro_usd=sum(value.max_cost_micro_usd for value in values),
        audit=_audit(),
    )


def test_attachment_plan_is_strict_versioned_and_partitions_artifacts() -> None:
    plan = _attachment_plan()

    assert tuple(work.work_key for work in plan.works) == ("group-a", "group-b")
    assert plan.max_parallel_groups == 2
    assert plan.to_ref().object_type == "attachment-generation-plan"

    payload = plan.model_dump(mode="python")
    payload["workspace_path"] = "/tmp/leak"
    with pytest.raises(ValidationError):
        AttachmentGenerationPlanV2.model_validate(payload)

    overlapping = (
        _attachment_work("group-a", artifact_ids=("artifact-a",)),
        _attachment_work("group-b", artifact_ids=("artifact-a",)),
    )
    with pytest.raises(ValidationError, match="exactly one attachment work"):
        _attachment_plan(works=overlapping)


def test_attachment_plan_requires_closed_acyclic_successors() -> None:
    cycle = (
        _attachment_work(
            "group-a",
            artifact_ids=("artifact-a",),
            dependencies=("group-b",),
        ),
        _attachment_work(
            "group-b",
            artifact_ids=("artifact-b",),
            dependencies=("group-a",),
        ),
    )
    with pytest.raises(ValidationError, match="cycle"):
        _attachment_plan(works=cycle)

    first = _attachment_plan()
    successor = AttachmentGenerationPlanV2.create(
        plan_id=first.plan_id,
        run_ref=first.run_ref,
        plan_version=2,
        predecessor_plan_ref=first.to_ref(),
        producer_task_view_ref=first.producer_task_view_ref,
        evidence_bundle_ref=first.evidence_bundle_ref,
        attachment_planning_context_ref=first.attachment_planning_context_ref,
        works=first.works,
        max_parallel_groups=first.max_parallel_groups,
        quality_policy_ref=first.quality_policy_ref,
        solvability_policy_ref=first.solvability_policy_ref,
        total_model_requests=first.total_model_requests,
        total_model_tokens=first.total_model_tokens,
        total_cost_micro_usd=first.total_cost_micro_usd,
        audit=_audit("successor"),
    )
    assert successor.predecessor_plan_ref == first.to_ref()

    with pytest.raises(ValidationError, match="predecessor"):
        AttachmentGenerationPlanV2.create(
            plan_id=first.plan_id,
            run_ref=first.run_ref,
            plan_version=2,
            predecessor_plan_ref=None,
            producer_task_view_ref=first.producer_task_view_ref,
            evidence_bundle_ref=first.evidence_bundle_ref,
            attachment_planning_context_ref=first.attachment_planning_context_ref,
            works=first.works,
            max_parallel_groups=first.max_parallel_groups,
            quality_policy_ref=first.quality_policy_ref,
            solvability_policy_ref=first.solvability_policy_ref,
            total_model_requests=first.total_model_requests,
            total_model_tokens=first.total_model_tokens,
            total_cost_micro_usd=first.total_cost_micro_usd,
            audit=_audit(),
        )


def test_attachment_results_have_exact_partitions_and_safe_assessments() -> None:
    plan = _attachment_plan()
    group_result = AttachmentGroupResultV2.create(
        result_id="attachment-group-result://group-a",
        plan_ref=plan.to_ref(),
        work_key="work-a",
        artifact_group_ref=plan.works[0].artifact_group_ref,
        execution_batch_ref=_ref("artifact-execution-batch"),
        reconstruction_result_ref=_ref("attachment-reconstruction-result"),
        execution_receipt_refs=(_ref("artifact-execution-receipt"),),
        artifact_ids=plan.works[0].artifact_ids,
        outcome=AgentTaskOutcomeV2.SUCCEEDED,
        reason_codes=(),
        audit=_audit(),
    )
    succeeded = _ref("agent-result-envelope", "group-a")
    blocked = _ref("agent-result-envelope", "group-b")
    subgraph = AttachmentSubgraphResultV2.create(
        result_id="attachment-subgraph-result://example",
        plan_ref=plan.to_ref(),
        work_result_refs=_sorted_refs(succeeded, blocked),
        succeeded_work_result_refs=(succeeded,),
        retryable_work_result_refs=(),
        blocked_work_result_refs=(blocked,),
        cancelled_work_result_refs=(),
        outcome=AttachmentSubgraphOutcomeV2.PARTIAL,
        reason_codes=("GROUP_BLOCKED",),
        audit=_audit(),
    )
    quality = AttachmentQualityAssessmentV2.create(
        assessment_id="attachment-quality-assessment://example",
        attachment_subgraph_result_ref=subgraph.to_ref(),
        item_quality_result_ref=_ref("item-quality-compilation-result"),
        validator_result_refs=(_ref("validator-result"),),
        outcome=AttachmentQualityOutcomeV2.BLOCKED,
        nonwaivable=False,
        reason_codes=("GROUP_BLOCKED",),
        audit=_audit(),
    )
    solvability = SolvabilityAssessmentV2.create(
        assessment_id="solvability-assessment://example",
        attachment_subgraph_result_ref=subgraph.to_ref(),
        quality_assessment_ref=quality.to_ref(),
        route_decision_ref=_ref("model-route-decision"),
        gateway_receipt_ref=_ref("gateway-receipt"),
        evidence_refs=(_ref("solvability-evidence"),),
        outcome=SolvabilityOutcomeV2.BLOCKED,
        reason_codes=("QUALITY_BLOCKED",),
        audit=_audit(),
    )

    assert group_result.artifact_ids == plan.works[0].artifact_ids
    assert solvability.quality_assessment_ref == quality.to_ref()
    with pytest.raises(
        ValidationError,
        match="execution receipts",
    ):
        AttachmentGroupResultV2.create(
            result_id="attachment-group-result://bad-receipt",
            plan_ref=plan.to_ref(),
            work_key="work-a",
            artifact_group_ref=plan.works[0].artifact_group_ref,
            execution_batch_ref=_ref("artifact-execution-batch"),
            reconstruction_result_ref=_ref("attachment-reconstruction-result"),
            execution_receipt_refs=(_ref("validator-result"),),
            artifact_ids=plan.works[0].artifact_ids,
            outcome=AgentTaskOutcomeV2.SUCCEEDED,
            reason_codes=(),
            audit=_audit(),
        )
    with pytest.raises(
        ValidationError,
        match="requires reasons",
    ):
        AttachmentGroupResultV2.create(
            result_id="attachment-group-result://missing-reason",
            plan_ref=plan.to_ref(),
            work_key="work-a",
            artifact_group_ref=plan.works[0].artifact_group_ref,
            execution_batch_ref=_ref("artifact-execution-batch"),
            reconstruction_result_ref=_ref("attachment-reconstruction-result"),
            execution_receipt_refs=(_ref("artifact-execution-receipt"),),
            artifact_ids=plan.works[0].artifact_ids,
            outcome=AgentTaskOutcomeV2.BLOCKED,
            reason_codes=(),
            audit=_audit(),
        )
    with pytest.raises(ValidationError, match="exact partition"):
        AttachmentSubgraphResultV2.create(
            result_id="attachment-subgraph-result://bad-partition",
            plan_ref=plan.to_ref(),
            work_result_refs=_sorted_refs(succeeded, blocked),
            succeeded_work_result_refs=(succeeded,),
            retryable_work_result_refs=(),
            blocked_work_result_refs=(),
            cancelled_work_result_refs=(),
            outcome=AttachmentSubgraphOutcomeV2.SUCCEEDED,
            reason_codes=(),
            audit=_audit(),
        )

    forbidden = {
        "grader_rule",
        "model_output_body",
        "private_reference",
        "prompt_body",
        "workspace_path",
    }
    for model_type in (
        AttachmentGenerationPlanV2,
        AttachmentGroupResultV2,
        AttachmentMockWorkV2,
        AttachmentQualityAssessmentV2,
        AttachmentSubgraphResultV2,
        SolvabilityAssessmentV2,
    ):
        assert forbidden.isdisjoint(model_type.model_fields)
