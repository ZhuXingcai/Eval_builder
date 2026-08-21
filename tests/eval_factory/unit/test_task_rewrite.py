from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts import (
    TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2,
    EvaluatorExecutionModeV2,
    PromptLeakageCategoryV2,
    R4TaskContractSetV2,
    RubricCriterionV2,
    RubricJudgedObjectKindV2,
    RubricJudgedObjectV2,
    RubricReachabilityV2,
    RubricSetV2,
    RubricSourceModeV2,
    TaskDraftPromptSafetyStatusV2,
    TaskDraftV2,
    TaskPromptSafetyGateStatusV2,
    TaskRequirementLineageV2,
    TaskRewriteRebuildStageV2,
    evaluator_spec_ref,
    producer_task_view_ref,
    r4_task_contract_set_ref,
    reference_policy_ref,
    rubric_criterion_carried_sha256,
    rubric_reachability_carried_sha256,
    rubric_set_carried_sha256,
    rubric_set_ref,
    task_draft_carried_sha256,
    task_draft_ref,
    task_prompt_safety_gate_ref,
    task_rewrite_application_ref,
    tool_policy_ref,
)
from eval_factory.contracts.approval import (
    ExampleKind,
    PlanExample,
    RewriteFidelity,
    TaskRewritePlan,
    TypedAdjustment,
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
    FakeTaskPromptSafetyFixture,
    FakeTaskPromptSafetyRunner,
    FakeTaskRewritePromptFixture,
    FakeTaskRewritePromptRunner,
    ProducerTaskViewCompiler,
    ProducerTaskViewProjectionOutcome,
    R4TaskContractSetCompiler,
    TaskRewriteAdjustmentCompiler,
    TaskRewriteApplicationCompiler,
    TaskRewritePlanCompileOutcome,
    TaskRewritePlanCompiler,
    TaskRewritePolicyError,
    TaskRewritePromptCompiler,
    TaskRewritePromptOutcome,
    TaskRewritePromptReason,
    TaskRewritePromptRequestBuilder,
    ToolCapabilityCatalog,
    ToolCapabilityDefinition,
    ToolPolicyCompilationOutcome,
    ToolPolicyCompiler,
    tool_capability_catalog_carried_sha256,
    tool_capability_definition_carried_sha256,
)
from eval_factory.task_authoring.prompt_safety import (
    PromptLeakageReferenceSetCompiler,
    TaskPromptSafetyCompiler,
    TaskPromptSafetyRequestBuilder,
)
from eval_factory.task_authoring.prompt_safety_models import (
    TaskPromptSafetyFindingFixture,
    TaskPromptSafetyOutcome,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64
SOURCE_TRACE_ID = "source-trace://task-rewrite/r4-09"
TRACE_ID = "trace-ir://task-rewrite/r4-09"
REQUIREMENT_ID = "requirement://task-rewrite/visible"
DEPENDENCY_ID = "attachment-dependency://task-rewrite/workspace"
EVALUATOR_BINDING_ID = "evaluator-binding://task-rewrite/default"
PRODUCER_PRINCIPAL = "principal://attachment-producer/task-rewrite"


def _audit(
    created_at: datetime = datetime(2026, 7, 27, tzinfo=UTC),
    *,
    input_refs: tuple[ObjectRef, ...] = (),
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="task-rewrite-test",
        governing_versions=(VersionBinding(component="task-rewrite", version="r4-09-v1"),),
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
        span_id=f"source-span://task-rewrite/{suffix}",
        source_trace_id=SOURCE_TRACE_ID,
        raw_sha256=HASH,
    )


def _evidence(
    suffix: str,
    *,
    subject_ref: ObjectRef | None = None,
) -> EvidenceRef:
    return EvidenceRef(
        evidence_ref_id=f"evidence-ref://task-rewrite/{suffix}",
        subject_ref=subject_ref or _ref("file-version-projection", suffix),
        source_spans=(_span(suffix),),
        polarity=EvidencePolarity.POSITIVE,
        capability="task-rewrite",
        capability_complete=True,
    )


def _dependency() -> AttachmentDependency:
    return AttachmentDependency(
        dependency_id=DEPENDENCY_ID,
        description="The input-state workspace required by the visible task.",
        criticality=AttachmentCriticality.REQUIRED,
        evidence_priority=EvidencePriority.EXPLICIT_REQUIREMENT,
        evidence=(_evidence("dependency"),),
    )


def _lineage() -> TaskRequirementLineageV2:
    return TaskRequirementLineageV2(
        requirement_id=REQUIREMENT_ID,
        statement="Inspect the input-state workspace and explain its design.",
        criticality=AttachmentCriticality.CRITICAL,
        evidence_priority=EvidencePriority.EXPLICIT_REQUIREMENT,
        evidence=(_evidence("requirement"),),
        task_episode_refs=(_ref("task-episode", "primary", version="v2"),),
        conflict_status=RequirementConflict.NONE,
    )


def _pending_draft(
    *,
    visible_prompt: str = ("Inspect the provided input-state workspace and explain its design."),
    task_version: int = 1,
    predecessor: ObjectRef | None = None,
    model_profile: str = "internal-task-author-v1",
    prompt_version: str = "task-draft-authoring-prompt/v1",
    policy_version: str = "task-draft-authoring/r4-03-v1",
    audit: ContractAudit | None = None,
) -> TaskDraftV2:
    draft = TaskDraftV2(
        task_draft_id="task-draft://pending",
        task_version=task_version,
        supersedes_task_draft_ref=predecessor,
        selection_context_ref=_ref(
            "selection-context",
            "task-rewrite",
            version="v2",
        ),
        task_episode_refs=(_ref("task-episode", "primary", version="v2"),),
        visible_prompt=visible_prompt,
        task_intent="Evaluate evidence-grounded workspace analysis.",
        evaluation_claim=("The contestant must infer the workspace design from safe inputs."),
        required_capabilities=("workspace-analysis",),
        allowed_tools=("file-read",),
        forbidden_outputs=(
            "original final answer",
            "completed deliverable",
            "private reference",
            "grader rule",
            "hidden pass condition",
            "secret",
        ),
        attachment_dependencies=(_dependency(),),
        requirement_lineage=(_lineage(),),
        prompt_requirement_ids=(REQUIREMENT_ID,),
        uncertainties=(),
        prompt_safety_status=TaskDraftPromptSafetyStatusV2.PENDING,
        prompt_safety_gate_ref=None,
        model_profile=model_profile,
        prompt_version=prompt_version,
        policy_version=policy_version,
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


def _reference_set():
    return PromptLeakageReferenceSetCompiler().compile(
        trace_envelope_ref=_ref("trace-envelope", TRACE_ID),
        sources=(),
        complete_categories=TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2,
        audit=_audit(),
    )


def _pass_prompt_safety(
    pending: TaskDraftV2,
    *,
    reference_set=None,
):
    resolved_reference_set = reference_set or _reference_set()
    request = TaskPromptSafetyRequestBuilder().build(
        task_draft=pending,
        leakage_reference_set=resolved_reference_set,
        model_profile="internal-task-safety-v1",
        prompt_version="task-safety-prompt/v1",
        audit=_audit(),
    )
    proposal = FakeTaskPromptSafetyRunner().run(
        request,
        fixture=FakeTaskPromptSafetyFixture(
            fixture_id="fixture://task-rewrite/safety-clear",
            outcome=TaskPromptSafetyOutcome.PASSED,
        ),
        audit=_audit(),
    )
    result = TaskPromptSafetyCompiler().compile(
        request=request,
        proposal=proposal,
        task_draft=pending,
        leakage_reference_set=resolved_reference_set,
        audit=_audit(),
    )
    assert result.outcome is TaskPromptSafetyOutcome.PASSED
    assert result.task_draft is not None
    assert result.task_prompt_safety_gate is not None
    return (
        result.task_draft,
        result.task_prompt_safety_gate,
        resolved_reference_set,
    )


def _criterion() -> RubricCriterionV2:
    reachability = RubricReachabilityV2(
        reachability_id="rubric-reachability://pending",
        prompt_requirement_ids=(REQUIREMENT_ID,),
        attachment_dependency_ids=(DEPENDENCY_ID,),
        allowed_tool_ids=("file-read",),
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
            judged_object_id="judged-object://task-rewrite/workspace",
            kind=RubricJudgedObjectKindV2.WORKSPACE_STATE,
            description="The observable workspace design state.",
        ),
        description="Correctly explains the workspace design.",
        weight=1.0,
        reachability=reachability,
        visibility=RubricVisibility.EVALUATOR_ONLY,
        evaluator_binding=EVALUATOR_BINDING_ID,
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


def _rubric_set(task_draft: TaskDraftV2) -> RubricSetV2:
    criterion = _criterion()
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
        audit=_audit(),
    )
    digest = rubric_set_carried_sha256(rubric)
    return rubric.model_copy(
        update={
            "rubric_set_id": f"rubric-set://sha256/{digest}",
            "rubric_set_sha256": digest,
        }
    )


def _evaluation_contract(rubric_set: RubricSetV2):
    result = EvaluationContractCompiler().compile(
        rubric_set=rubric_set,
        binding_definitions=(
            EvaluatorBindingDefinition(
                evaluator_binding_id=EVALUATOR_BINDING_ID,
                evaluator_type="contract-evaluator",
                execution_mode=EvaluatorExecutionModeV2.DETERMINISTIC,
                evaluator_version="r4-06-v1",
                input_contract_ref=_ref(
                    "evaluator-input-contract",
                    "task-rewrite",
                ),
                output_contract_ref=_ref(
                    "evaluator-output-contract",
                    "task-rewrite",
                ),
                evaluator_principal_id=("principal://evaluator/task-rewrite"),
                reference_refs=(),
                timeout_seconds=300,
            ),
        ),
        reference_mode=ReferenceMode.NONE,
        audit=_audit(),
    )
    assert result.outcome is EvaluationContractOutcome.COMPILED
    assert result.evaluator_spec is not None
    assert result.reference_policy is not None
    return result.evaluator_spec, result.reference_policy


def _tool_definition() -> ToolCapabilityDefinition:
    definition = ToolCapabilityDefinition(
        definition_id="tool-capability-definition://pending",
        tool_id="file-read",
        tool_family=ToolFamily.FILE_READ,
        capability_ref=_ref("tool-capability", "file-read"),
        enforcement_profile_ref=_ref(
            "tool-enforcement-profile",
            "file-read",
        ),
        constraint_profile_ref=_ref(
            "tool-constraint-profile",
            "file-read",
        ),
        contestant_descriptor_ref=_ref(
            "contestant-tool-descriptor",
            "file-read",
        ),
        contestant_constraint_profile_ref=_ref(
            "contestant-tool-constraint-profile",
            "file-read",
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


def _tool_catalog() -> ToolCapabilityCatalog:
    catalog = ToolCapabilityCatalog(
        catalog_id="tool-capability-catalog://pending",
        catalog_version="tool-capability-catalog/r4-07-v1",
        definitions=(_tool_definition(),),
        catalog_sha256=HASH,
        audit=_audit(),
    )
    digest = tool_capability_catalog_carried_sha256(catalog)
    return catalog.model_copy(
        update={
            "catalog_id": (f"tool-capability-catalog://sha256/{digest}"),
            "catalog_sha256": digest,
        }
    )


def _tool_contract(
    task_draft: TaskDraftV2,
    rubric_set: RubricSetV2,
    evaluator_spec,
    catalog: ToolCapabilityCatalog,
):
    result = ToolPolicyCompiler().compile(
        task_draft=task_draft,
        rubric_set=rubric_set,
        evaluator_spec=evaluator_spec,
        catalog=catalog,
        audit=_audit(),
    )
    assert result.outcome is ToolPolicyCompilationOutcome.COMPILED
    assert result.tool_policy is not None
    assert result.contestant_projection is not None
    return result.tool_policy, result.contestant_projection


def _producer_decision(subject: ObjectRef) -> ProvenanceDecision:
    return ProvenanceDecision(
        provenance_decision_id="provenance-decision://task-rewrite/input",
        subject_ref=subject,
        origin_class=OriginClass.PREEXISTING_WORKSPACE_INPUT,
        visibility=Visibility.STAGE_PROJECTION,
        disposition=Disposition.ALLOW_INPUT_EVIDENCE,
        rule_ids=("task-rewrite-producer/test-v1",),
        source_event_refs=(_ref("trace-event", "task-rewrite/input"),),
        confidence=1.0,
        review_required=False,
        policy_version="task-rewrite-producer/test-v1",
        subject_sha256=subject.object_sha256,
        audit=_audit(),
    )


def _producer_evidence():
    source_ref = _ref("file-version", "task-rewrite/input")
    view_result = EvidenceViewEngine().project(
        EvidenceViewRequest(
            principal=EvidenceViewPrincipal(
                principal_id=PRODUCER_PRINCIPAL,
                principal_type=(EvidenceViewPrincipalType.ATTACHMENT_PRODUCER),
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
    bundle = (
        EvidenceBundleCompiler()
        .compile(
            EvidenceBundleCompileRequest(
                source_trace_id=SOURCE_TRACE_ID,
                trace_ir_version_id=TRACE_ID,
                consumer_stage="attachment-producer",
                purpose="attachment-production",
                view_result=view_result,
                bindings=(
                    ProjectionEvidenceBinding(
                        projection_item_id=(view_result.included_items[0].projection_item_id),
                        source_spans=(_span("producer-bundle"),),
                        polarity=EvidencePolarity.POSITIVE,
                        capability="attachment-production",
                        capability_complete=True,
                    ),
                ),
                max_characters=1000,
                audit=_audit(),
            )
        )
        .evidence_bundle
    )
    return view_result, bundle


def _source_chain():
    pending = _pending_draft()
    task_draft, gate, reference_set = _pass_prompt_safety(pending)
    rubric_set = _rubric_set(task_draft)
    evaluator_spec, reference_policy = _evaluation_contract(rubric_set)
    catalog = _tool_catalog()
    tool_policy, contestant_policy = _tool_contract(
        task_draft,
        rubric_set,
        evaluator_spec,
        catalog,
    )
    producer_view_result, producer_bundle = _producer_evidence()
    producer = ProducerTaskViewCompiler().compile(
        task_draft=task_draft,
        rubric_set=rubric_set,
        evaluator_spec=evaluator_spec,
        reference_policy=reference_policy,
        tool_policy=tool_policy,
        contestant_tool_policy=contestant_policy,
        producer_view_result=producer_view_result,
        producer_evidence_bundle=producer_bundle,
        audit=_audit(),
    )
    assert producer.outcome is ProducerTaskViewProjectionOutcome.PROJECTED
    assert producer.producer_task_view is not None
    assert producer.storage_authorization is not None
    return {
        "pending": pending,
        "task_draft": task_draft,
        "gate": gate,
        "reference_set": reference_set,
        "rubric_set": rubric_set,
        "evaluator_spec": evaluator_spec,
        "reference_policy": reference_policy,
        "catalog": catalog,
        "tool_policy": tool_policy,
        "contestant_policy": contestant_policy,
        "producer_view_result": producer_view_result,
        "producer_bundle": producer_bundle,
        "storage_authorization": producer.storage_authorization,
        "producer_task_view": producer.producer_task_view,
    }


def _compile_contract_set(chain: dict[str, object]) -> R4TaskContractSetV2:
    return R4TaskContractSetCompiler().compile(
        task_draft=chain["task_draft"],
        task_prompt_safety_gate=chain["gate"],
        rubric_set=chain["rubric_set"],
        evaluator_spec=chain["evaluator_spec"],
        reference_policy=chain["reference_policy"],
        tool_policy=chain["tool_policy"],
        contestant_tool_policy=chain["contestant_policy"],
        producer_storage_authorization=chain["storage_authorization"],
        producer_task_view=chain["producer_task_view"],
        audit=_audit(),
    )


def _plan(chain: dict[str, object], **overrides: object) -> TaskRewritePlan:
    task_draft = chain["task_draft"]
    assert isinstance(task_draft, TaskDraftV2)
    values: dict[str, object] = {
        "task_rewrite_plan_id": "task-rewrite-plan://candidate",
        "selection_context_ref": task_draft.selection_context_ref,
        "target_capability": task_draft.evaluation_claim,
        "rewrite_style": "Concise, self-contained engineering task.",
        "fidelity": RewriteFidelity.CAPABILITY_PRESERVING,
        "operational_noise_policy": "Remove retries and runtime chatter.",
        "examples": (
            PlanExample(
                example_id="plan-example://task-rewrite/primary",
                kind=ExampleKind.REWRITE,
                input_summary="A multi-turn workspace analysis request.",
                expected_treatment=("Inspect the workspace and explain its design."),
                evidence_refs=(task_draft.requirement_lineage[0].evidence[0],),
            ),
        ),
        "forbidden_content_rules": task_draft.forbidden_outputs,
        "expected_capability_impact": "Preserve workspace-analysis capability.",
        "audit": _audit(),
    }
    values.update(overrides)
    return TaskRewritePlan(**values)


def _clear_safety_proposal(
    request,
    *,
    audit: ContractAudit | None = None,
):
    return FakeTaskPromptSafetyRunner().run(
        request,
        fixture=FakeTaskPromptSafetyFixture(
            fixture_id="fixture://task-rewrite/preview-clear",
            outcome=TaskPromptSafetyOutcome.PASSED,
        ),
        audit=audit or _audit(),
    )


def _compile_plan(
    chain: dict[str, object],
    *,
    audit: ContractAudit | None = None,
):
    resolved_audit = audit or _audit()
    contract_set = _compile_contract_set(chain)
    compiler = TaskRewritePlanCompiler()
    plan = _plan(chain)
    request = compiler.build_safety_request(
        plan=plan,
        contract_set=contract_set,
        task_draft=chain["task_draft"],
        leakage_reference_set=chain["reference_set"],
        model_profile="internal-task-rewrite-safety-v1",
        prompt_version="task-rewrite-safety-prompt/v1",
        audit=resolved_audit,
    )
    result = compiler.compile(
        plan=plan,
        contract_set=contract_set,
        task_draft=chain["task_draft"],
        task_prompt_safety_gate=chain["gate"],
        rubric_set=chain["rubric_set"],
        evaluator_spec=chain["evaluator_spec"],
        reference_policy=chain["reference_policy"],
        tool_policy=chain["tool_policy"],
        contestant_tool_policy=chain["contestant_policy"],
        producer_storage_authorization=chain["storage_authorization"],
        producer_task_view=chain["producer_task_view"],
        leakage_reference_set=chain["reference_set"],
        safety_request=request,
        safety_proposal=_clear_safety_proposal(
            request,
            audit=resolved_audit,
        ),
        audit=resolved_audit,
    )
    return contract_set, result


def test_contract_set_compiler_validates_exact_passed_r4_chain() -> None:
    chain = _source_chain()
    contract_set = _compile_contract_set(chain)

    assert contract_set.task_draft_ref == task_draft_ref(chain["task_draft"])
    assert contract_set.task_prompt_safety_gate_ref == (task_prompt_safety_gate_ref(chain["gate"]))
    assert contract_set.producer_task_view_ref == producer_task_view_ref(chain["producer_task_view"])

    stale_gate = chain["gate"].model_copy(update={"gate_sha256": OTHER_HASH})
    with pytest.raises(TaskRewritePolicyError, match="prompt safety"):
        R4TaskContractSetCompiler().compile(
            task_draft=chain["task_draft"],
            task_prompt_safety_gate=stale_gate,
            rubric_set=chain["rubric_set"],
            evaluator_spec=chain["evaluator_spec"],
            reference_policy=chain["reference_policy"],
            tool_policy=chain["tool_policy"],
            contestant_tool_policy=chain["contestant_policy"],
            producer_storage_authorization=chain["storage_authorization"],
            producer_task_view=chain["producer_task_view"],
            audit=_audit(),
        )


def test_plan_compiler_emits_safe_versioned_preview() -> None:
    chain = _source_chain()
    contract_set, result = _compile_plan(chain)

    assert result.outcome is TaskRewritePlanCompileOutcome.PREVIEWED
    assert result.plan_version is not None
    assert result.preview_safety_gate is not None
    assert result.preview is not None
    assert result.plan_version.plan_version == 1
    assert result.plan_version.basis_contract_set_ref == (r4_task_contract_set_ref(contract_set))
    assert result.preview.examples[0].example_id == (result.plan_version.plan.examples[0].example_id)
    serialized = str(result.preview.model_dump(mode="json"))
    assert "evidence_refs" not in serialized
    assert "private-reference://" not in serialized
    assert "rubric-set://" not in serialized


def test_plan_compiler_rejects_hidden_capability_and_unsafe_evidence() -> None:
    chain = _source_chain()
    contract_set = _compile_contract_set(chain)
    compiler = TaskRewritePlanCompiler()

    wrong_capability = _plan(chain, target_capability="hidden label")
    with pytest.raises(TaskRewritePolicyError, match="target capability"):
        compiler.build_safety_request(
            plan=wrong_capability,
            contract_set=contract_set,
            task_draft=chain["task_draft"],
            leakage_reference_set=chain["reference_set"],
            model_profile="internal-task-rewrite-safety-v1",
            prompt_version="task-rewrite-safety-prompt/v1",
            audit=_audit(),
        )

    unsafe_example = (
        _plan(chain)
        .examples[0]
        .model_copy(
            update={
                "evidence_refs": (
                    _evidence(
                        "unsafe",
                        subject_ref=_ref("private-reference", "unsafe"),
                    ),
                )
            }
        )
    )
    with pytest.raises(TaskRewritePolicyError, match="evidence"):
        compiler.build_safety_request(
            plan=_plan(chain, examples=(unsafe_example,)),
            contract_set=contract_set,
            task_draft=chain["task_draft"],
            leakage_reference_set=chain["reference_set"],
            model_profile="internal-task-rewrite-safety-v1",
            prompt_version="task-rewrite-safety-prompt/v1",
            audit=_audit(),
        )


def test_preview_safety_block_and_missing_model_emit_no_plan_or_preview() -> None:
    chain = _source_chain()
    contract_set = _compile_contract_set(chain)
    compiler = TaskRewritePlanCompiler()
    plan = _plan(chain)
    request = compiler.build_safety_request(
        plan=plan,
        contract_set=contract_set,
        task_draft=chain["task_draft"],
        leakage_reference_set=chain["reference_set"],
        model_profile="internal-task-rewrite-safety-v1",
        prompt_version="task-rewrite-safety-prompt/v1",
        audit=_audit(),
    )
    blocked_proposal = FakeTaskPromptSafetyRunner().run(
        request,
        fixture=FakeTaskPromptSafetyFixture(
            fixture_id="fixture://task-rewrite/preview-blocked",
            outcome=TaskPromptSafetyOutcome.BLOCKED,
            findings=(
                TaskPromptSafetyFindingFixture(
                    category=PromptLeakageCategoryV2.TRAJECTORY_SPECIFIC_STEP,
                    prompt_span_start=0,
                    prompt_span_end=8,
                    rule_id="task-rewrite-preview-rule://trajectory",
                ),
            ),
        ),
        audit=_audit(),
    )
    blocked = compiler.compile(
        plan=plan,
        contract_set=contract_set,
        task_draft=chain["task_draft"],
        task_prompt_safety_gate=chain["gate"],
        rubric_set=chain["rubric_set"],
        evaluator_spec=chain["evaluator_spec"],
        reference_policy=chain["reference_policy"],
        tool_policy=chain["tool_policy"],
        contestant_tool_policy=chain["contestant_policy"],
        producer_storage_authorization=chain["storage_authorization"],
        producer_task_view=chain["producer_task_view"],
        leakage_reference_set=chain["reference_set"],
        safety_request=request,
        safety_proposal=blocked_proposal,
        audit=_audit(),
    )
    assert blocked.outcome is TaskRewritePlanCompileOutcome.BLOCKED_SAFETY
    assert blocked.plan_version is None
    assert blocked.preview_safety_gate is not None
    assert blocked.preview_safety_gate.status is (TaskPromptSafetyGateStatusV2.BLOCKED)
    assert blocked.preview_safety_gate.findings
    assert blocked.preview is None

    missing = compiler.compile(
        plan=plan,
        contract_set=contract_set,
        task_draft=chain["task_draft"],
        task_prompt_safety_gate=chain["gate"],
        rubric_set=chain["rubric_set"],
        evaluator_spec=chain["evaluator_spec"],
        reference_policy=chain["reference_policy"],
        tool_policy=chain["tool_policy"],
        contestant_tool_policy=chain["contestant_policy"],
        producer_storage_authorization=chain["storage_authorization"],
        producer_task_view=chain["producer_task_view"],
        leakage_reference_set=chain["reference_set"],
        safety_request=request,
        safety_proposal=None,
        audit=_audit(),
    )
    assert missing.outcome is TaskRewritePlanCompileOutcome.BLOCKED_CAPABILITY
    assert missing.plan_version is None
    assert missing.preview_safety_gate is None
    assert missing.preview is None


def test_plan_preview_identities_ignore_audit_actor_time() -> None:
    chain = _source_chain()
    first_contract_set, first = _compile_plan(
        chain,
        audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)),
    )
    second_contract_set, second = _compile_plan(
        chain,
        audit=_audit(datetime(2026, 7, 28, tzinfo=UTC)),
    )
    assert first.plan_version is not None
    assert first.preview_safety_gate is not None
    assert first.preview is not None
    assert second.plan_version is not None
    assert second.preview_safety_gate is not None
    assert second.preview is not None
    assert first_contract_set.contract_set_sha256 == (second_contract_set.contract_set_sha256)
    assert first.plan_version.task_rewrite_plan_version_sha256 == (
        second.plan_version.task_rewrite_plan_version_sha256
    )
    assert first.preview_safety_gate.gate_sha256 == (second.preview_safety_gate.gate_sha256)
    assert first.preview.preview_sha256 == second.preview.preview_sha256
    assert first.preview.canonical_sha256() != second.preview.canonical_sha256()


def test_adjustment_creates_new_plan_and_exact_r4_invalidation() -> None:
    chain = _source_chain()
    contract_set, compiled = _compile_plan(chain)
    assert compiled.plan_version is not None
    assert compiled.preview is not None
    assert compiled.preview_safety_gate is not None
    compiler = TaskRewriteAdjustmentCompiler()
    adjustments = (
        TypedAdjustment(
            target_path="rewrite_style",
            operation="SET",
            value="Direct and concise.",
            reason="Use the configured task style.",
        ),
        TypedAdjustment(
            target_path=(
                "examples["
                + hashlib.sha256(b"plan-example://task-rewrite/primary").hexdigest()
                + "].expected_treatment"
            ),
            operation="REPLACE",
            value="Inspect the workspace and summarize its design.",
            reason="Clarify the example.",
        ),
    )
    request = compiler.build_safety_request(
        source_plan_version=compiled.plan_version,
        source_preview=compiled.preview,
        source_preview_safety_gate=compiled.preview_safety_gate,
        source_contract_set=contract_set,
        current_application=None,
        adjustments=adjustments,
        task_draft=chain["task_draft"],
        leakage_reference_set=chain["reference_set"],
        model_profile="internal-task-rewrite-safety-v1",
        prompt_version="task-rewrite-safety-prompt/v1",
        audit=_audit(),
    )
    result = compiler.apply(
        source_plan_version=compiled.plan_version,
        source_preview=compiled.preview,
        source_preview_safety_gate=compiled.preview_safety_gate,
        source_contract_set=contract_set,
        current_application=None,
        adjustments=adjustments,
        task_draft=chain["task_draft"],
        task_prompt_safety_gate=chain["gate"],
        rubric_set=chain["rubric_set"],
        evaluator_spec=chain["evaluator_spec"],
        reference_policy=chain["reference_policy"],
        tool_policy=chain["tool_policy"],
        contestant_tool_policy=chain["contestant_policy"],
        producer_storage_authorization=chain["storage_authorization"],
        producer_task_view=chain["producer_task_view"],
        leakage_reference_set=chain["reference_set"],
        safety_request=request,
        safety_proposal=_clear_safety_proposal(request),
        preserved_object_refs=(
            chain["task_draft"].selection_context_ref,
            *chain["task_draft"].task_episode_refs,
            ObjectRef(
                object_type="tool-capability-catalog",
                object_id=chain["catalog"].catalog_id,
                object_version=chain["catalog"].catalog_version,
                object_sha256=chain["catalog"].catalog_sha256,
            ),
        ),
        audit=_audit(),
    )

    assert result.replacement_plan_version is not None
    assert result.replacement_preview is not None
    assert result.invalidation is not None
    assert result.replacement_plan_version.plan_version == 2
    assert result.replacement_plan_version.plan.rewrite_style == ("Direct and concise.")
    assert result.invalidation.required_rebuild_stages == tuple(TaskRewriteRebuildStageV2)
    assert tuple(ref.object_type for ref in result.invalidation.invalidated_object_refs) == (
        "task-draft",
        "task-prompt-safety-gate",
        "rubric-set",
        "evaluator-spec",
        "reference-policy",
        "tool-policy",
        "contestant-tool-policy",
        "producer-storage-authorization",
        "producer-task-view",
    )


def test_adjustment_rejects_forbidden_path_noop_and_hard_gate_removal() -> None:
    chain = _source_chain()
    contract_set, compiled = _compile_plan(chain)
    assert compiled.plan_version is not None
    assert compiled.preview is not None
    assert compiled.preview_safety_gate is not None
    compiler = TaskRewriteAdjustmentCompiler()

    for adjustment in (
        TypedAdjustment(
            target_path="target_capability",
            operation="SET",
            value="different capability",
            reason="Forbidden.",
        ),
        TypedAdjustment(
            target_path="rewrite_style",
            operation="SET",
            value=compiled.plan_version.plan.rewrite_style,
            reason="No-op.",
        ),
        TypedAdjustment(
            target_path="forbidden_content_rules",
            operation="REMOVE",
            value="original final answer",
            reason="Forbidden weakening.",
        ),
    ):
        with pytest.raises(TaskRewritePolicyError):
            compiler.build_safety_request(
                source_plan_version=compiled.plan_version,
                source_preview=compiled.preview,
                source_preview_safety_gate=compiled.preview_safety_gate,
                source_contract_set=contract_set,
                current_application=None,
                adjustments=(adjustment,),
                task_draft=chain["task_draft"],
                leakage_reference_set=chain["reference_set"],
                model_profile="internal-task-rewrite-safety-v1",
                prompt_version="task-rewrite-safety-prompt/v1",
                audit=_audit(),
            )

    duplicate = TypedAdjustment(
        target_path="rewrite_style",
        operation="SET",
        value="A different valid style.",
        reason="Duplicate target.",
    )
    with pytest.raises(TaskRewritePolicyError, match="unique"):
        compiler.build_safety_request(
            source_plan_version=compiled.plan_version,
            source_preview=compiled.preview,
            source_preview_safety_gate=compiled.preview_safety_gate,
            source_contract_set=contract_set,
            current_application=None,
            adjustments=(duplicate, duplicate),
            task_draft=chain["task_draft"],
            leakage_reference_set=chain["reference_set"],
            model_profile="internal-task-rewrite-safety-v1",
            prompt_version="task-rewrite-safety-prompt/v1",
            audit=_audit(),
        )


def _adjusted_plan(chain: dict[str, object]):
    contract_set, compiled = _compile_plan(chain)
    assert compiled.plan_version is not None
    assert compiled.preview is not None
    assert compiled.preview_safety_gate is not None
    compiler = TaskRewriteAdjustmentCompiler()
    adjustments = (
        TypedAdjustment(
            target_path="rewrite_style",
            operation="SET",
            value="Direct and concise.",
            reason="Use the configured task style.",
        ),
    )
    safety_request = compiler.build_safety_request(
        source_plan_version=compiled.plan_version,
        source_preview=compiled.preview,
        source_preview_safety_gate=compiled.preview_safety_gate,
        source_contract_set=contract_set,
        current_application=None,
        adjustments=adjustments,
        task_draft=chain["task_draft"],
        leakage_reference_set=chain["reference_set"],
        model_profile="internal-task-rewrite-safety-v1",
        prompt_version="task-rewrite-safety-prompt/v1",
        audit=_audit(),
    )
    result = compiler.apply(
        source_plan_version=compiled.plan_version,
        source_preview=compiled.preview,
        source_preview_safety_gate=compiled.preview_safety_gate,
        source_contract_set=contract_set,
        current_application=None,
        adjustments=adjustments,
        task_draft=chain["task_draft"],
        task_prompt_safety_gate=chain["gate"],
        rubric_set=chain["rubric_set"],
        evaluator_spec=chain["evaluator_spec"],
        reference_policy=chain["reference_policy"],
        tool_policy=chain["tool_policy"],
        contestant_tool_policy=chain["contestant_policy"],
        producer_storage_authorization=chain["storage_authorization"],
        producer_task_view=chain["producer_task_view"],
        leakage_reference_set=chain["reference_set"],
        safety_request=safety_request,
        safety_proposal=_clear_safety_proposal(safety_request),
        preserved_object_refs=(
            chain["task_draft"].selection_context_ref,
            *chain["task_draft"].task_episode_refs,
            ObjectRef(
                object_type="tool-capability-catalog",
                object_id=chain["catalog"].catalog_id,
                object_version=chain["catalog"].catalog_version,
                object_sha256=chain["catalog"].catalog_sha256,
            ),
        ),
        audit=_audit(),
    )
    assert result.replacement_plan_version is not None
    assert result.replacement_preview is not None
    assert result.replacement_preview_safety_gate is not None
    assert result.invalidation is not None
    return contract_set, compiled, result


def test_prompt_rewrite_emits_only_pending_structural_successor() -> None:
    chain = _source_chain()
    _, _, adjusted = _adjusted_plan(chain)
    builder = TaskRewritePromptRequestBuilder()
    request = builder.build(
        source_task_draft=chain["task_draft"],
        replacement_plan_version=adjusted.replacement_plan_version,
        replacement_preview=adjusted.replacement_preview,
        replacement_preview_safety_gate=(adjusted.replacement_preview_safety_gate),
        model_profile="internal-task-rewriter-v1",
        prompt_version="task-rewrite-prompt/v1",
        audit=_audit(),
    )
    proposal = FakeTaskRewritePromptRunner().run(
        request,
        fixture=FakeTaskRewritePromptFixture(
            fixture_id="fixture://task-rewrite/prompt",
            outcome=TaskRewritePromptOutcome.REWRITTEN,
            visible_prompt=("Inspect the input-state workspace and summarize its design."),
            covered_prompt_requirement_ids=(REQUIREMENT_ID,),
        ),
        audit=_audit(),
    )
    result = TaskRewritePromptCompiler().compile(
        request=request,
        proposal=proposal,
        source_task_draft=chain["task_draft"],
        replacement_plan_version=adjusted.replacement_plan_version,
        replacement_preview=adjusted.replacement_preview,
        replacement_preview_safety_gate=(adjusted.replacement_preview_safety_gate),
        audit=_audit(),
    )

    assert result.outcome is TaskRewritePromptOutcome.REWRITTEN
    assert result.task_draft is not None
    rewritten = result.task_draft
    assert rewritten.prompt_safety_status is (TaskDraftPromptSafetyStatusV2.PENDING)
    assert rewritten.task_version == chain["task_draft"].task_version + 1
    assert rewritten.supersedes_task_draft_ref == task_draft_ref(chain["task_draft"])
    assert rewritten.visible_prompt != chain["task_draft"].visible_prompt
    for field in (
        "selection_context_ref",
        "task_episode_refs",
        "task_intent",
        "evaluation_claim",
        "required_capabilities",
        "allowed_tools",
        "forbidden_outputs",
        "attachment_dependencies",
        "requirement_lineage",
        "prompt_requirement_ids",
        "uncertainties",
    ):
        assert getattr(rewritten, field) == getattr(
            chain["task_draft"],
            field,
        )


def test_prompt_rewrite_blocks_missing_model_and_incomplete_coverage() -> None:
    chain = _source_chain()
    _, _, adjusted = _adjusted_plan(chain)
    request = TaskRewritePromptRequestBuilder().build(
        source_task_draft=chain["task_draft"],
        replacement_plan_version=adjusted.replacement_plan_version,
        replacement_preview=adjusted.replacement_preview,
        replacement_preview_safety_gate=(adjusted.replacement_preview_safety_gate),
        model_profile="internal-task-rewriter-v1",
        prompt_version="task-rewrite-prompt/v1",
        audit=_audit(),
    )
    missing = TaskRewritePromptCompiler().compile(
        request=request,
        proposal=None,
        source_task_draft=chain["task_draft"],
        replacement_plan_version=adjusted.replacement_plan_version,
        replacement_preview=adjusted.replacement_preview,
        replacement_preview_safety_gate=(adjusted.replacement_preview_safety_gate),
        audit=_audit(),
    )
    assert missing.outcome is TaskRewritePromptOutcome.BLOCKED_CAPABILITY
    assert missing.task_draft is None
    assert missing.unresolved_reasons == frozenset({TaskRewritePromptReason.MODEL_UNAVAILABLE})

    incomplete_proposal = FakeTaskRewritePromptRunner().run(
        request,
        fixture=FakeTaskRewritePromptFixture(
            fixture_id="fixture://task-rewrite/incomplete-coverage",
            outcome=TaskRewritePromptOutcome.REWRITTEN,
            visible_prompt="Inspect the workspace and summarize its design.",
            covered_prompt_requirement_ids=("requirement://task-rewrite/unrelated",),
        ),
        audit=_audit(),
    )
    with pytest.raises(TaskRewritePolicyError, match="coverage"):
        TaskRewritePromptCompiler().compile(
            request=request,
            proposal=incomplete_proposal,
            source_task_draft=chain["task_draft"],
            replacement_plan_version=adjusted.replacement_plan_version,
            replacement_preview=adjusted.replacement_preview,
            replacement_preview_safety_gate=(adjusted.replacement_preview_safety_gate),
            audit=_audit(),
        )


def test_standard_environment_rewrite_is_typed_blocked() -> None:
    chain = _source_chain()
    contract_set, compiled = _compile_plan(chain)
    assert compiled.plan_version is not None
    assert compiled.preview is not None
    compiler = TaskRewriteAdjustmentCompiler()
    adjustment = TypedAdjustment(
        target_path="fidelity",
        operation="SET",
        value="STANDARD_ENVIRONMENT",
        reason="Use a standard environment.",
    )
    request = compiler.build_safety_request(
        source_plan_version=compiled.plan_version,
        source_preview=compiled.preview,
        source_preview_safety_gate=compiled.preview_safety_gate,
        source_contract_set=contract_set,
        current_application=None,
        adjustments=(adjustment,),
        task_draft=chain["task_draft"],
        leakage_reference_set=chain["reference_set"],
        model_profile="internal-task-rewrite-safety-v1",
        prompt_version="task-rewrite-safety-prompt/v1",
        audit=_audit(),
    )
    adjusted = compiler.apply(
        source_plan_version=compiled.plan_version,
        source_preview=compiled.preview,
        source_preview_safety_gate=compiled.preview_safety_gate,
        source_contract_set=contract_set,
        current_application=None,
        adjustments=(adjustment,),
        task_draft=chain["task_draft"],
        task_prompt_safety_gate=chain["gate"],
        rubric_set=chain["rubric_set"],
        evaluator_spec=chain["evaluator_spec"],
        reference_policy=chain["reference_policy"],
        tool_policy=chain["tool_policy"],
        contestant_tool_policy=chain["contestant_policy"],
        producer_storage_authorization=chain["storage_authorization"],
        producer_task_view=chain["producer_task_view"],
        leakage_reference_set=chain["reference_set"],
        safety_request=request,
        safety_proposal=_clear_safety_proposal(request),
        preserved_object_refs=(
            chain["task_draft"].selection_context_ref,
            *chain["task_draft"].task_episode_refs,
            ObjectRef(
                object_type="tool-capability-catalog",
                object_id=chain["catalog"].catalog_id,
                object_version=chain["catalog"].catalog_version,
                object_sha256=chain["catalog"].catalog_sha256,
            ),
        ),
        audit=_audit(),
    )
    assert adjusted.replacement_plan_version is not None
    assert adjusted.replacement_preview is not None
    assert adjusted.replacement_preview_safety_gate is not None
    request = TaskRewritePromptRequestBuilder().build(
        source_task_draft=chain["task_draft"],
        replacement_plan_version=adjusted.replacement_plan_version,
        replacement_preview=adjusted.replacement_preview,
        replacement_preview_safety_gate=(adjusted.replacement_preview_safety_gate),
        model_profile="internal-task-rewriter-v1",
        prompt_version="task-rewrite-prompt/v1",
        audit=_audit(),
    )
    result = TaskRewritePromptCompiler().compile(
        request=request,
        proposal=None,
        source_task_draft=chain["task_draft"],
        replacement_plan_version=adjusted.replacement_plan_version,
        replacement_preview=adjusted.replacement_preview,
        replacement_preview_safety_gate=(adjusted.replacement_preview_safety_gate),
        audit=_audit(),
    )
    assert result.outcome is TaskRewritePromptOutcome.BLOCKED_CAPABILITY
    assert result.task_draft is None
    assert result.unresolved_reasons == frozenset({TaskRewritePromptReason.ENVIRONMENT_STRATEGY_REQUIRED})


def test_application_compiler_builds_complete_successor_chain() -> None:
    chain = _source_chain()
    source_contract_set, _, adjusted = _adjusted_plan(chain)
    request = TaskRewritePromptRequestBuilder().build(
        source_task_draft=chain["task_draft"],
        replacement_plan_version=adjusted.replacement_plan_version,
        replacement_preview=adjusted.replacement_preview,
        replacement_preview_safety_gate=(adjusted.replacement_preview_safety_gate),
        model_profile="internal-task-rewriter-v1",
        prompt_version="task-rewrite-prompt/v1",
        audit=_audit(),
    )
    proposal = FakeTaskRewritePromptRunner().run(
        request,
        fixture=FakeTaskRewritePromptFixture(
            fixture_id="fixture://task-rewrite/application",
            outcome=TaskRewritePromptOutcome.REWRITTEN,
            visible_prompt=("Inspect the input-state workspace and summarize its design."),
            covered_prompt_requirement_ids=(REQUIREMENT_ID,),
        ),
        audit=_audit(),
    )
    rewritten = TaskRewritePromptCompiler().compile(
        request=request,
        proposal=proposal,
        source_task_draft=chain["task_draft"],
        replacement_plan_version=adjusted.replacement_plan_version,
        replacement_preview=adjusted.replacement_preview,
        replacement_preview_safety_gate=(adjusted.replacement_preview_safety_gate),
        audit=_audit(),
    )
    assert rewritten.task_draft is not None
    passed_task, passed_gate, _ = _pass_prompt_safety(
        rewritten.task_draft,
        reference_set=chain["reference_set"],
    )
    compiler = TaskRewriteApplicationCompiler()
    application_inputs = {
        "source_contract_set": source_contract_set,
        "source_plan_version": adjusted.source_plan_version,
        "replacement_plan_version": adjusted.replacement_plan_version,
        "invalidation": adjusted.invalidation,
        "source_task_draft": chain["task_draft"],
        "source_task_prompt_safety_gate": chain["gate"],
        "source_rubric_set": chain["rubric_set"],
        "source_evaluator_spec": chain["evaluator_spec"],
        "source_reference_policy": chain["reference_policy"],
        "source_tool_policy": chain["tool_policy"],
        "source_contestant_tool_policy": chain["contestant_policy"],
        "source_storage_authorization": chain["storage_authorization"],
        "source_producer_task_view": chain["producer_task_view"],
        "rewritten_pending_task_draft": rewritten.task_draft,
        "rewritten_passed_task_draft": passed_task,
        "rewritten_task_prompt_safety_gate": passed_gate,
        "tool_catalog": chain["catalog"],
        "producer_view_result": chain["producer_view_result"],
        "producer_evidence_bundle": chain["producer_bundle"],
    }
    result = compiler.compile(
        **application_inputs,
        audit=_audit(),
    )

    assert result.application is not None
    assert result.replacement_contract_set is not None
    assert result.rubric_set is not None
    assert result.evaluator_spec is not None
    assert result.reference_policy is not None
    assert result.tool_policy is not None
    assert result.contestant_tool_policy is not None
    assert result.storage_authorization is not None
    assert result.producer_task_view is not None
    assert result.rubric_set.rubric_version == (chain["rubric_set"].rubric_version + 1)
    assert result.rubric_set.supersedes_rubric_set_ref == rubric_set_ref(chain["rubric_set"])
    assert result.evaluator_spec.evaluator_spec_version == (
        chain["evaluator_spec"].evaluator_spec_version + 1
    )
    assert result.evaluator_spec.supersedes_evaluator_spec_ref == (
        evaluator_spec_ref(chain["evaluator_spec"])
    )
    assert result.reference_policy.reference_policy_version == (
        chain["reference_policy"].reference_policy_version + 1
    )
    assert result.reference_policy.supersedes_reference_policy_ref == (
        reference_policy_ref(chain["reference_policy"])
    )
    assert result.tool_policy.tool_policy_version == (chain["tool_policy"].tool_policy_version + 1)
    assert result.tool_policy.supersedes_tool_policy_ref == tool_policy_ref(chain["tool_policy"])
    assert result.producer_task_view.producer_task_view_version == (
        chain["producer_task_view"].producer_task_view_version + 1
    )
    assert result.producer_task_view.supersedes_producer_task_view_ref == (
        producer_task_view_ref(chain["producer_task_view"])
    )
    assert task_rewrite_application_ref(result.application).object_type == ("task-rewrite-application")
    assert result.application.replacement_contract_set_ref == (
        r4_task_contract_set_ref(result.replacement_contract_set)
    )
    compiler.validate_current(
        application_result=result,
        **application_inputs,
    )

    next_adjustment = TypedAdjustment(
        target_path="expected_capability_impact",
        operation="SET",
        value="Preserve workspace analysis with a clearer prompt.",
        reason="Clarify the capability impact.",
    )
    adjustment_compiler = TaskRewriteAdjustmentCompiler()
    next_safety_request = adjustment_compiler.build_safety_request(
        source_plan_version=adjusted.replacement_plan_version,
        source_preview=adjusted.replacement_preview,
        source_preview_safety_gate=(adjusted.replacement_preview_safety_gate),
        source_contract_set=result.replacement_contract_set,
        current_application=result.application,
        adjustments=(next_adjustment,),
        task_draft=passed_task,
        leakage_reference_set=chain["reference_set"],
        model_profile="internal-task-rewrite-safety-v1",
        prompt_version="task-rewrite-safety-prompt/v1",
        audit=_audit(),
    )
    next_adjusted = adjustment_compiler.apply(
        source_plan_version=adjusted.replacement_plan_version,
        source_preview=adjusted.replacement_preview,
        source_preview_safety_gate=(adjusted.replacement_preview_safety_gate),
        source_contract_set=result.replacement_contract_set,
        current_application=result.application,
        adjustments=(next_adjustment,),
        task_draft=passed_task,
        task_prompt_safety_gate=passed_gate,
        rubric_set=result.rubric_set,
        evaluator_spec=result.evaluator_spec,
        reference_policy=result.reference_policy,
        tool_policy=result.tool_policy,
        contestant_tool_policy=result.contestant_tool_policy,
        producer_storage_authorization=result.storage_authorization,
        producer_task_view=result.producer_task_view,
        leakage_reference_set=chain["reference_set"],
        safety_request=next_safety_request,
        safety_proposal=_clear_safety_proposal(next_safety_request),
        preserved_object_refs=(
            passed_task.selection_context_ref,
            *passed_task.task_episode_refs,
            ObjectRef(
                object_type="tool-capability-catalog",
                object_id=chain["catalog"].catalog_id,
                object_version=chain["catalog"].catalog_version,
                object_sha256=chain["catalog"].catalog_sha256,
            ),
        ),
        audit=_audit(),
    )
    assert next_adjusted.replacement_plan_version is not None
    assert next_adjusted.replacement_plan_version.plan_version == 3

    with pytest.raises(TaskRewritePolicyError, match="current"):
        compiler.validate_current(
            application_result=result.model_copy(update={"result_sha256": OTHER_HASH}),
            **application_inputs,
        )


def test_rewrite_models_reject_authority_fields_from_semantic_fixture() -> None:
    with pytest.raises(ValidationError):
        FakeTaskRewritePromptFixture.model_validate(
            {
                "fixture_id": "fixture://task-rewrite/authority",
                "outcome": "REWRITTEN",
                "visible_prompt": "Safe prompt.",
                "covered_prompt_requirement_ids": [REQUIREMENT_ID],
                "task_draft_ref": _ref(
                    "task-draft",
                    "forbidden",
                    version="v2",
                ).model_dump(mode="json"),
            }
        )
