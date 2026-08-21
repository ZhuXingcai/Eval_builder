from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts import (
    EvaluatorExecutionModeV2,
    ProducerStorageAuthorizationV2,
    RubricCriterionV2,
    RubricJudgedObjectKindV2,
    RubricJudgedObjectV2,
    RubricReachabilityV2,
    RubricSetV2,
    RubricSourceModeV2,
    TaskDraftPromptSafetyStatusV2,
    TaskDraftV2,
    TaskRequirementLineageV2,
    producer_storage_authorization_carried_sha256,
    producer_storage_authorization_ref,
    producer_task_view_carried_sha256,
    rubric_criterion_carried_sha256,
    rubric_reachability_carried_sha256,
    rubric_set_carried_sha256,
    task_draft_carried_sha256,
    task_draft_ref,
)
from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    EvidenceRef,
    ObjectRef,
    SourceSpanRef,
    VersionBinding,
)
from eval_factory.contracts.safety import (
    Disposition,
    OriginClass,
    ProvenanceDecision,
    Visibility,
)
from eval_factory.contracts.task import (
    AttachmentCriticality,
    AttachmentDependency,
    EvidencePriority,
    ReferenceMode,
    RequirementConflict,
    RubricVisibility,
)
from eval_factory.contracts.trace import ToolFamily
from eval_factory.provenance import (
    EvidenceBundleCompiler,
    EvidenceBundleCompileRequest,
    EvidenceViewEngine,
    EvidenceViewPrincipal,
    EvidenceViewPrincipalType,
    EvidenceViewPurpose,
    EvidenceViewRequest,
    EvidenceViewSubject,
    ProjectionEvidenceBinding,
)
from eval_factory.task_authoring import (
    EvaluationContractCompiler,
    EvaluationContractOutcome,
    EvaluatorBindingDefinition,
    ProducerStorageAccessGate,
    ProducerStorageAccessOutcome,
    ProducerStorageAccessReason,
    ProducerStorageAccessRequest,
    ProducerTaskViewCompiler,
    ProducerTaskViewPolicyError,
    ProducerTaskViewProjectionOutcome,
    ProducerTaskViewProjectionReason,
    ToolCapabilityCatalog,
    ToolCapabilityDefinition,
    ToolPolicyCompilationOutcome,
    ToolPolicyCompiler,
    producer_storage_access_request_carried_sha256,
    tool_capability_catalog_carried_sha256,
    tool_capability_definition_carried_sha256,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64
THIRD_HASH = "c" * 64
FOURTH_HASH = "d" * 64
SOURCE_TRACE_ID = "source-trace://producer/r4-08"
TRACE_ID = "trace-ir://producer/r4-08"
DEPENDENCY_ID = "attachment-dependency://producer/workspace"
REQUIREMENT_ID = "requirement://producer/visible"
EVALUATOR_BINDING = "evaluator-binding://producer/default"
PRODUCER_PRINCIPAL = "principal://attachment-producer/r4-08"
ROOT = Path(__file__).resolve().parents[3]


def _audit(
    created_at: datetime = datetime(2026, 7, 27, tzinfo=UTC),
    *,
    input_refs: tuple[ObjectRef, ...] = (),
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="producer-task-view-test",
        governing_versions=(VersionBinding(component="producer-task-view", version="r4-08"),),
        input_refs=input_refs,
    )


def _ref(
    object_type: str,
    suffix: str,
    *,
    digest: str = HASH,
    version: str = "v1",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _span(suffix: str) -> SourceSpanRef:
    return SourceSpanRef(
        span_id=f"source-span://producer/{suffix}",
        source_trace_id=SOURCE_TRACE_ID,
        raw_sha256=HASH,
    )


def _evidence(
    suffix: str,
    *,
    subject_ref: ObjectRef | None = None,
) -> EvidenceRef:
    return EvidenceRef(
        evidence_ref_id=f"evidence-ref://producer/{suffix}",
        subject_ref=subject_ref or _ref("file-version-projection", suffix),
        source_spans=(_span(suffix),),
        polarity=EvidencePolarity.POSITIVE,
        capability="producer-task-view",
        capability_complete=True,
    )


def _dependency() -> AttachmentDependency:
    return AttachmentDependency(
        dependency_id=DEPENDENCY_ID,
        description="The visible input-state workspace required by the task.",
        criticality=AttachmentCriticality.REQUIRED,
        evidence_priority=EvidencePriority.EXPLICIT_REQUIREMENT,
        evidence=(_evidence("dependency"),),
    )


def _lineage() -> TaskRequirementLineageV2:
    return TaskRequirementLineageV2(
        requirement_id=REQUIREMENT_ID,
        statement="Inspect the visible input-state workspace.",
        criticality=AttachmentCriticality.CRITICAL,
        evidence_priority=EvidencePriority.EXPLICIT_REQUIREMENT,
        evidence=(_evidence("requirement"),),
        task_episode_refs=(_ref("task-episode", "primary", version="v2"),),
        conflict_status=RequirementConflict.NONE,
    )


def _draft(
    *,
    status: TaskDraftPromptSafetyStatusV2 = TaskDraftPromptSafetyStatusV2.PASSED,
    allowed_tools: tuple[str, ...] = ("file-read",),
    audit: ContractAudit | None = None,
) -> TaskDraftV2:
    gate_ref = (
        None
        if status is TaskDraftPromptSafetyStatusV2.PENDING
        else _ref(
            "task-prompt-safety-gate",
            "producer",
            version="v2",
        )
    )
    draft = TaskDraftV2(
        task_draft_id="task-draft://pending",
        task_version=1,
        supersedes_task_draft_ref=None,
        selection_context_ref=_ref(
            "selection-context",
            "producer",
            version="v2",
        ),
        task_episode_refs=(_ref("task-episode", "primary", version="v2"),),
        visible_prompt=("Inspect the provided input-state workspace using only allowed tools."),
        task_intent="Evaluate evidence-grounded workspace inspection.",
        evaluation_claim="The response is derivable from visible input-state facts.",
        required_capabilities=("workspace-inspection",),
        allowed_tools=allowed_tools,
        forbidden_outputs=(
            "original final answer",
            "private grader controls",
        ),
        attachment_dependencies=(_dependency(),),
        requirement_lineage=(_lineage(),),
        prompt_requirement_ids=(REQUIREMENT_ID,),
        uncertainties=(),
        prompt_safety_status=status,
        prompt_safety_gate_ref=gate_ref,
        model_profile="internal-task-author-v1",
        prompt_version="task-draft-authoring-prompt/v1",
        policy_version="task-draft-authoring/r4-03-v1",
        task_draft_sha256=HASH,
        audit=audit or _audit(),
    )
    digest = task_draft_carried_sha256(draft)
    return draft.model_copy(
        update={
            "task_draft_id": f"task-draft://sha256/{digest}",
            "task_draft_sha256": digest,
        }
    )


def _criterion(
    *,
    allowed_tool_ids: tuple[str, ...] = ("file-read",),
) -> RubricCriterionV2:
    reachability = RubricReachabilityV2(
        reachability_id="rubric-reachability://pending",
        prompt_requirement_ids=(REQUIREMENT_ID,),
        attachment_dependency_ids=(DEPENDENCY_ID,),
        allowed_tool_ids=allowed_tool_ids,
        evidence=(_evidence("rubric"),),
        capability_complete=True,
        reachability_sha256=HASH,
    )
    reachability_digest = rubric_reachability_carried_sha256(reachability)
    reachability = reachability.model_copy(
        update={
            "reachability_id": (f"rubric-reachability://sha256/{reachability_digest}"),
            "reachability_sha256": reachability_digest,
        }
    )
    criterion = RubricCriterionV2(
        criterion_id="rubric-criterion://pending",
        judged_object=RubricJudgedObjectV2(
            judged_object_id="judged-object://producer/workspace",
            kind=RubricJudgedObjectKindV2.WORKSPACE_STATE,
            description="The workspace state produced for evaluation.",
        ),
        description="The workspace satisfies the visible requirement.",
        weight=1.0,
        reachability=reachability,
        visibility=RubricVisibility.EVALUATOR_ONLY,
        evaluator_binding=EVALUATOR_BINDING,
        approval_status="CANDIDATE",
        criterion_sha256=HASH,
    )
    criterion_digest = rubric_criterion_carried_sha256(criterion)
    return criterion.model_copy(
        update={
            "criterion_id": f"rubric-criterion://sha256/{criterion_digest}",
            "criterion_sha256": criterion_digest,
        }
    )


def _rubric_set(
    task_draft: TaskDraftV2,
    *,
    audit: ContractAudit | None = None,
) -> RubricSetV2:
    criterion = _criterion(allowed_tool_ids=task_draft.allowed_tools)
    rubric = RubricSetV2(
        rubric_set_id="rubric-set://pending",
        rubric_version=1,
        supersedes_rubric_set_ref=None,
        task_draft_ref=task_draft_ref(task_draft),
        source_mode=RubricSourceModeV2.GENERATED,
        criteria=(criterion,),
        total_weight=criterion.weight,
        imported_from_ref=None,
        model_profile="internal-rubric-author-v1",
        prompt_version="rubric-authoring-prompt/v1",
        policy_version="rubric-authoring/r4-05-v1",
        rubric_set_sha256=HASH,
        audit=audit or _audit(),
    )
    digest = rubric_set_carried_sha256(rubric)
    return rubric.model_copy(
        update={
            "rubric_set_id": f"rubric-set://sha256/{digest}",
            "rubric_set_sha256": digest,
        }
    )


def _evaluation_contract(
    rubric_set: RubricSetV2,
    *,
    reference_mode: ReferenceMode = ReferenceMode.NONE,
):
    reference_refs = (
        ()
        if reference_mode is ReferenceMode.NONE
        else (
            _ref(
                {
                    ReferenceMode.PRIVATE_ANSWER: "private-reference",
                    ReferenceMode.STRUCTURED_EXPECTATIONS: ("structured-expectation"),
                    ReferenceMode.TRACE_BEHAVIOR: "trace-behavior-reference",
                    ReferenceMode.HUMAN_ONLY: "human-only-reference",
                }[reference_mode],
                "producer/hidden",
                digest=FOURTH_HASH,
            ),
        )
    )
    execution_mode = (
        EvaluatorExecutionModeV2.HUMAN_ONLY
        if reference_mode is ReferenceMode.HUMAN_ONLY
        else EvaluatorExecutionModeV2.DETERMINISTIC
    )
    result = EvaluationContractCompiler().compile(
        rubric_set=rubric_set,
        binding_definitions=(
            EvaluatorBindingDefinition(
                evaluator_binding_id=EVALUATOR_BINDING,
                evaluator_type="contract-evaluator",
                execution_mode=execution_mode,
                evaluator_version="r4-06-v1",
                input_contract_ref=_ref(
                    "evaluator-input-contract",
                    "producer",
                ),
                output_contract_ref=_ref(
                    "evaluator-output-contract",
                    "producer",
                ),
                evaluator_principal_id=(
                    None
                    if execution_mode is EvaluatorExecutionModeV2.HUMAN_ONLY
                    else "principal://evaluator/producer"
                ),
                model_profile_ref=None,
                reference_refs=reference_refs,
                timeout_seconds=300,
            ),
        ),
        reference_mode=reference_mode,
        audit=_audit(),
    )
    assert result.outcome is EvaluationContractOutcome.COMPILED
    assert result.evaluator_spec is not None
    assert result.reference_policy is not None
    return result.evaluator_spec, result.reference_policy


def _tool_definition(tool_id: str = "file-read") -> ToolCapabilityDefinition:
    definition = ToolCapabilityDefinition(
        definition_id="tool-capability-definition://pending",
        tool_id=tool_id,
        tool_family=ToolFamily.FILE_READ,
        capability_ref=_ref("tool-capability", tool_id),
        enforcement_profile_ref=_ref("tool-enforcement-profile", tool_id),
        constraint_profile_ref=_ref("tool-constraint-profile", tool_id),
        contestant_descriptor_ref=_ref(
            "contestant-tool-descriptor",
            tool_id,
        ),
        contestant_constraint_profile_ref=_ref(
            "contestant-tool-constraint-profile",
            tool_id,
        ),
        contestant_eligible=True,
        definition_sha256=HASH,
    )
    digest = tool_capability_definition_carried_sha256(definition)
    return definition.model_copy(
        update={
            "definition_id": (f"tool-capability-definition://sha256/{digest}"),
            "definition_sha256": digest,
        }
    )


def _tool_catalog(
    definitions: tuple[ToolCapabilityDefinition, ...] | None = None,
) -> ToolCapabilityCatalog:
    catalog = ToolCapabilityCatalog(
        catalog_id="tool-capability-catalog://pending",
        catalog_version="tool-capability-catalog/r4-07-v1",
        definitions=definitions or (_tool_definition(),),
        catalog_sha256=HASH,
        audit=_audit(),
    )
    digest = tool_capability_catalog_carried_sha256(catalog)
    return catalog.model_copy(
        update={
            "catalog_id": f"tool-capability-catalog://sha256/{digest}",
            "catalog_sha256": digest,
        }
    )


def _tool_contract(
    task_draft: TaskDraftV2,
    rubric_set: RubricSetV2,
    evaluator_spec,
):
    result = ToolPolicyCompiler().compile(
        task_draft=task_draft,
        rubric_set=rubric_set,
        evaluator_spec=evaluator_spec,
        catalog=_tool_catalog(),
        audit=_audit(),
    )
    assert result.outcome is ToolPolicyCompilationOutcome.COMPILED
    assert result.tool_policy is not None
    assert result.contestant_projection is not None
    return result.tool_policy, result.contestant_projection


def _producer_decision(subject: ObjectRef) -> ProvenanceDecision:
    return ProvenanceDecision(
        provenance_decision_id="provenance-decision://producer/input",
        subject_ref=subject,
        origin_class=OriginClass.PREEXISTING_WORKSPACE_INPUT,
        visibility=Visibility.STAGE_PROJECTION,
        disposition=Disposition.ALLOW_INPUT_EVIDENCE,
        rule_ids=("producer-view-parent/v1",),
        source_event_refs=(_ref("trace-event", "producer/input"),),
        confidence=1.0,
        review_required=False,
        policy_version="producer-view-parent/test-v1",
        subject_sha256=subject.object_sha256,
        audit=_audit(),
    )


def _producer_evidence(
    *,
    empty: bool = False,
    principal_id: str = PRODUCER_PRINCIPAL,
):
    source_ref = _ref("file-version", "producer/input")
    view_result = EvidenceViewEngine().project(
        EvidenceViewRequest(
            principal=EvidenceViewPrincipal(
                principal_id=principal_id,
                principal_type=EvidenceViewPrincipalType.ATTACHMENT_PRODUCER,
                allowed_purposes=frozenset({EvidenceViewPurpose.ATTACHMENT_PRODUCTION}),
                max_subjects=10,
                max_characters=1000,
            ),
            purpose=EvidenceViewPurpose.ATTACHMENT_PRODUCTION,
            subjects=(
                EvidenceViewSubject(
                    subject_ref=source_ref,
                    decision=_producer_decision(source_ref),
                    projection_text="safe input-state content",
                ),
            ),
            max_characters=1000,
            audit=_audit(),
        )
    )
    bindings = (
        ()
        if empty
        else (
            ProjectionEvidenceBinding(
                projection_item_id=(view_result.included_items[0].projection_item_id),
                source_spans=(_span("bundle"),),
                polarity=EvidencePolarity.POSITIVE,
                capability="attachment-production",
                capability_complete=True,
            ),
        )
    )
    bundle_result = EvidenceBundleCompiler().compile(
        EvidenceBundleCompileRequest(
            source_trace_id=SOURCE_TRACE_ID,
            trace_ir_version_id=TRACE_ID,
            consumer_stage="attachment-producer",
            purpose="attachment-production",
            view_result=view_result,
            bindings=bindings,
            max_characters=1000,
            audit=_audit(),
        )
    )
    return view_result, bundle_result.evidence_bundle


def _inputs(
    *,
    status: TaskDraftPromptSafetyStatusV2 = TaskDraftPromptSafetyStatusV2.PASSED,
    empty_evidence: bool = False,
    reference_mode: ReferenceMode = ReferenceMode.NONE,
    audit: ContractAudit | None = None,
):
    task_draft = _draft(status=status, audit=audit)
    chain_draft = (
        _draft(status=TaskDraftPromptSafetyStatusV2.PASSED, audit=audit)
        if status is TaskDraftPromptSafetyStatusV2.BLOCKED
        else task_draft
    )
    rubric_set = _rubric_set(chain_draft, audit=audit)
    evaluator_spec, reference_policy = _evaluation_contract(
        rubric_set,
        reference_mode=reference_mode,
    )
    tool_policy, contestant_policy = _tool_contract(
        chain_draft,
        rubric_set,
        evaluator_spec,
    )
    producer_view, producer_bundle = _producer_evidence(empty=empty_evidence)
    return (
        task_draft,
        rubric_set,
        evaluator_spec,
        reference_policy,
        tool_policy,
        contestant_policy,
        producer_view,
        producer_bundle,
    )


def _compile(
    *,
    status: TaskDraftPromptSafetyStatusV2 = TaskDraftPromptSafetyStatusV2.PASSED,
    empty_evidence: bool = False,
    reference_mode: ReferenceMode = ReferenceMode.NONE,
    audit: ContractAudit | None = None,
):
    inputs = _inputs(
        status=status,
        empty_evidence=empty_evidence,
        reference_mode=reference_mode,
        audit=audit,
    )
    result = ProducerTaskViewCompiler().compile(
        task_draft=inputs[0],
        rubric_set=inputs[1],
        evaluator_spec=inputs[2],
        reference_policy=inputs[3],
        tool_policy=inputs[4],
        contestant_tool_policy=inputs[5],
        producer_view_result=inputs[6],
        producer_evidence_bundle=inputs[7],
        audit=audit or _audit(),
    )
    return (*inputs, result)


def _access_request(
    authorization: ProducerStorageAuthorizationV2,
    *,
    principal_id: str = PRODUCER_PRINCIPAL,
    purpose: str = "ATTACHMENT_PRODUCTION",
    requested_subject_ref: ObjectRef | None = None,
) -> ProducerStorageAccessRequest:
    request = ProducerStorageAccessRequest(
        request_id="producer-storage-access-request://pending",
        authorization_ref=producer_storage_authorization_ref(authorization),
        producer_principal_id=principal_id,
        purpose=purpose,
        requested_subject_ref=(requested_subject_ref or authorization.authorized_subject_refs[0]),
        policy_version="producer-task-view/r4-08-v1",
        request_sha256=HASH,
        audit=_audit(),
    )
    digest = producer_storage_access_request_carried_sha256(request)
    return request.model_copy(
        update={
            "request_id": (f"producer-storage-access-request://sha256/{digest}"),
            "request_sha256": digest,
        }
    )


def test_compiler_projects_exact_visible_fields_and_authorized_subjects() -> None:
    (
        task_draft,
        _,
        _,
        _,
        _,
        contestant_policy,
        _,
        bundle,
        result,
    ) = _compile()

    assert result.outcome is ProducerTaskViewProjectionOutcome.PROJECTED
    assert result.producer_task_view is not None
    assert result.storage_authorization is not None
    assert result.unresolved_reasons == frozenset()
    view = result.producer_task_view
    authorization = result.storage_authorization
    assert view.query_instruction == task_draft.visible_prompt
    assert view.allowed_tools == ("file-read",)
    assert view.forbidden_outputs == task_draft.forbidden_outputs
    assert tuple(item.dependency_id for item in view.attachment_requirements) == (DEPENDENCY_ID,)
    assert set(type(view.attachment_requirements[0]).model_fields) == {
        "schema_version",
        "dependency_id",
        "description",
        "criticality",
    }
    assert authorization.authorized_subject_refs == tuple(item.subject_ref for item in bundle.evidence)
    assert authorization.authorization_sha256 == (
        producer_storage_authorization_carried_sha256(authorization)
    )
    assert view.source_task_draft_sha256 == task_draft.task_draft_sha256
    assert view.source_contestant_tool_policy_sha256 == (contestant_policy.contestant_tool_policy_sha256)
    assert view.producer_task_view_sha256 == producer_task_view_carried_sha256(view)


def test_projection_excludes_evaluation_lineage_private_refs_and_caller_audit() -> None:
    caller_ref = _ref("private-reference", "caller-audit")
    *_, result = _compile(
        reference_mode=ReferenceMode.PRIVATE_ANSWER,
        audit=_audit(input_refs=(caller_ref,)),
    )
    assert result.producer_task_view is not None
    assert result.storage_authorization is not None

    serialized = str(
        {
            "view": result.producer_task_view.model_dump(mode="json"),
            "authorization": result.storage_authorization.model_dump(mode="json"),
        }
    )
    for forbidden in (
        "private-reference://producer/hidden",
        "private-reference://caller-audit",
        "rubric-set://",
        "evaluator-spec://",
        "reference-policy://",
        "tool-capability-catalog://",
        "tool-enforcement-profile://",
        "selection-context://",
        "task-episode://",
        "source-span://",
        "provenance-decision://",
        "safe input-state content",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        pytest.param(
            TaskDraftPromptSafetyStatusV2.PENDING,
            ProducerTaskViewProjectionReason.PROMPT_SAFETY_PENDING,
            id="pending",
        ),
        pytest.param(
            TaskDraftPromptSafetyStatusV2.BLOCKED,
            ProducerTaskViewProjectionReason.PROMPT_SAFETY_BLOCKED,
            id="blocked",
        ),
    ],
)
def test_nonpassed_task_is_typed_blocked_without_partial_contracts(
    status: TaskDraftPromptSafetyStatusV2,
    reason: ProducerTaskViewProjectionReason,
) -> None:
    *_, result = _compile(status=status)

    assert result.outcome is ProducerTaskViewProjectionOutcome.BLOCKED_SAFETY
    assert result.producer_task_view is None
    assert result.storage_authorization is None
    assert result.unresolved_reasons == frozenset({reason})


def test_compiler_rejects_cross_chain_and_stale_nested_inputs() -> None:
    inputs = _inputs()
    compiler = ProducerTaskViewCompiler()

    other_draft = _draft(allowed_tools=())
    with pytest.raises(ProducerTaskViewPolicyError, match="TaskDraft"):
        compiler.compile(
            task_draft=other_draft,
            rubric_set=inputs[1],
            evaluator_spec=inputs[2],
            reference_policy=inputs[3],
            tool_policy=inputs[4],
            contestant_tool_policy=inputs[5],
            producer_view_result=inputs[6],
            producer_evidence_bundle=inputs[7],
            audit=_audit(),
        )

    stale_policy = inputs[3].model_copy(update={"reference_policy_sha256": OTHER_HASH})
    with pytest.raises(ProducerTaskViewPolicyError, match="ReferencePolicy"):
        compiler.compile(
            task_draft=inputs[0],
            rubric_set=inputs[1],
            evaluator_spec=inputs[2],
            reference_policy=stale_policy,
            tool_policy=inputs[4],
            contestant_tool_policy=inputs[5],
            producer_view_result=inputs[6],
            producer_evidence_bundle=inputs[7],
            audit=_audit(),
        )


def test_compiler_rejects_wrong_producer_view_or_bundle_binding() -> None:
    inputs = _inputs()
    compiler = ProducerTaskViewCompiler()
    wrong_principal_view = inputs[6].model_copy(
        update={
            "principal_type": EvidenceViewPrincipalType.EVALUATOR,
            "purpose": EvidenceViewPurpose.EVALUATION,
        }
    )
    with pytest.raises(ProducerTaskViewPolicyError, match="producer evidence"):
        compiler.compile(
            task_draft=inputs[0],
            rubric_set=inputs[1],
            evaluator_spec=inputs[2],
            reference_policy=inputs[3],
            tool_policy=inputs[4],
            contestant_tool_policy=inputs[5],
            producer_view_result=wrong_principal_view,
            producer_evidence_bundle=inputs[7],
            audit=_audit(),
        )

    wrong_bundle = inputs[7].model_copy(update={"purpose": "evaluation"})
    with pytest.raises(ProducerTaskViewPolicyError, match="producer evidence"):
        compiler.compile(
            task_draft=inputs[0],
            rubric_set=inputs[1],
            evaluator_spec=inputs[2],
            reference_policy=inputs[3],
            tool_policy=inputs[4],
            contestant_tool_policy=inputs[5],
            producer_view_result=inputs[6],
            producer_evidence_bundle=wrong_bundle,
            audit=_audit(),
        )


def test_empty_safe_bundle_remains_valid_prompt_only_handoff() -> None:
    *_, bundle, result = _compile(empty_evidence=True)

    assert bundle.evidence == ()
    assert result.outcome is ProducerTaskViewProjectionOutcome.PROJECTED
    assert result.producer_task_view is not None
    assert result.storage_authorization is not None
    assert result.storage_authorization.authorized_subject_refs == ()
    assert result.producer_task_view.safe_evidence_bundle_refs


def test_storage_access_gate_grants_exact_projected_member_only() -> None:
    *_, result = _compile()
    assert result.storage_authorization is not None
    authorization = result.storage_authorization
    request = _access_request(authorization)

    access = ProducerStorageAccessGate().authorize(
        authorization=authorization,
        request=request,
        audit=_audit(),
    )

    assert access.outcome is ProducerStorageAccessOutcome.GRANTED
    assert access.authorized_subject_ref == request.requested_subject_ref
    assert access.reasons == frozenset()


def test_storage_access_denials_do_not_disclose_subject_or_inventory() -> None:
    *_, result = _compile()
    assert result.storage_authorization is not None
    authorization = result.storage_authorization
    wrong_identity = _access_request(
        authorization,
        principal_id="principal://attachment-producer/other",
    )
    denied_identity = ProducerStorageAccessGate().authorize(
        authorization=authorization,
        request=wrong_identity,
        audit=_audit(),
    )
    assert denied_identity.outcome is (ProducerStorageAccessOutcome.DENIED_IDENTITY)
    assert denied_identity.authorized_subject_ref is None
    assert ProducerStorageAccessReason.PRINCIPAL_MISMATCH in (denied_identity.reasons)

    wrong_purpose = _access_request(
        authorization,
        purpose="EVALUATION",
    )
    denied_purpose = ProducerStorageAccessGate().authorize(
        authorization=authorization,
        request=wrong_purpose,
        audit=_audit(),
    )
    assert denied_purpose.outcome is ProducerStorageAccessOutcome.DENIED_IDENTITY
    assert ProducerStorageAccessReason.PURPOSE_MISMATCH in (denied_purpose.reasons)

    private_ref = _ref(
        "private-reference",
        "producer/denied",
        digest=FOURTH_HASH,
    )
    wrong_scope = _access_request(
        authorization,
        requested_subject_ref=private_ref,
    )
    denied_scope = ProducerStorageAccessGate().authorize(
        authorization=authorization,
        request=wrong_scope,
        audit=_audit(),
    )
    assert denied_scope.outcome is ProducerStorageAccessOutcome.DENIED_SCOPE
    assert denied_scope.authorized_subject_ref is None
    serialized = str(denied_scope.model_dump(mode="json"))
    assert private_ref.object_id not in serialized
    assert authorization.authorized_subject_refs[0].object_id not in serialized


def test_storage_access_gate_rejects_stale_authorization_or_request() -> None:
    *_, result = _compile()
    assert result.storage_authorization is not None
    authorization = result.storage_authorization
    request = _access_request(authorization)

    with pytest.raises(ProducerTaskViewPolicyError, match="authorization"):
        ProducerStorageAccessGate().authorize(
            authorization=authorization.model_copy(update={"authorization_sha256": OTHER_HASH}),
            request=request,
            audit=_audit(),
        )
    with pytest.raises(ProducerTaskViewPolicyError, match="request"):
        ProducerStorageAccessGate().authorize(
            authorization=authorization,
            request=request.model_copy(update={"request_sha256": OTHER_HASH}),
            audit=_audit(),
        )
    with pytest.raises(ValidationError):
        ProducerStorageAccessRequest.model_validate(
            {
                **request.model_dump(mode="python"),
                "storage_path": "/restricted",
            }
        )


def test_validate_current_accepts_exact_and_rejects_view_or_scope_mutation() -> None:
    *inputs, result = _compile()
    assert result.producer_task_view is not None
    assert result.storage_authorization is not None
    compiler = ProducerTaskViewCompiler()

    compiler.validate_current(
        task_draft=inputs[0],
        rubric_set=inputs[1],
        evaluator_spec=inputs[2],
        reference_policy=inputs[3],
        tool_policy=inputs[4],
        contestant_tool_policy=inputs[5],
        producer_view_result=inputs[6],
        producer_evidence_bundle=inputs[7],
        producer_task_view=result.producer_task_view,
        storage_authorization=result.storage_authorization,
    )

    with pytest.raises(ProducerTaskViewPolicyError, match="current"):
        compiler.validate_current(
            task_draft=inputs[0],
            rubric_set=inputs[1],
            evaluator_spec=inputs[2],
            reference_policy=inputs[3],
            tool_policy=inputs[4],
            contestant_tool_policy=inputs[5],
            producer_view_result=inputs[6],
            producer_evidence_bundle=inputs[7],
            producer_task_view=result.producer_task_view.model_copy(update={"allowed_tools": ()}),
            storage_authorization=result.storage_authorization,
        )

    with pytest.raises(ProducerTaskViewPolicyError, match="current"):
        compiler.validate_current(
            task_draft=inputs[0],
            rubric_set=inputs[1],
            evaluator_spec=inputs[2],
            reference_policy=inputs[3],
            tool_policy=inputs[4],
            contestant_tool_policy=inputs[5],
            producer_view_result=inputs[6],
            producer_evidence_bundle=inputs[7],
            producer_task_view=result.producer_task_view,
            storage_authorization=result.storage_authorization.model_copy(
                update={"authorized_subject_refs": ()}
            ),
        )


def test_projection_is_stable_across_audit_timestamps() -> None:
    first = _compile(audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)))[-1]
    second = _compile(audit=_audit(datetime(2026, 7, 28, tzinfo=UTC)))[-1]
    assert first.producer_task_view is not None
    assert second.producer_task_view is not None
    assert first.storage_authorization is not None
    assert second.storage_authorization is not None

    assert first.producer_task_view.producer_task_view_id == (second.producer_task_view.producer_task_view_id)
    assert first.storage_authorization.authorization_id == (second.storage_authorization.authorization_id)
    assert first.result_id == second.result_id


def test_r4_08_identities_are_stable_across_python_hash_seed() -> None:
    code = """
import runpy

ns = runpy.run_path("tests/eval_factory/unit/test_producer_task_view.py")
result = ns["_compile"]()[-1]
view = result.producer_task_view
authorization = result.storage_authorization
assert view is not None and authorization is not None
request = ns["_access_request"](authorization)
access = ns["ProducerStorageAccessGate"]().authorize(
    authorization=authorization,
    request=request,
    audit=ns["_audit"](),
)
print(view.producer_task_view_id)
print(authorization.authorization_id)
print(result.result_id)
print(access.result_id)
"""
    outputs = []
    for seed in ("1", "99"):
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PYTHONHASHSEED": seed,
                "PYTHONPATH": "src",
            },
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(completed.stdout.strip())

    assert len(set(outputs)) == 1
