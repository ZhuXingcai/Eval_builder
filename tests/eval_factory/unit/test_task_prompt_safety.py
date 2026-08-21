from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.agent_system.r4_authoring_agent import (
    TaskPromptSafetyAgentInputV1,
)
from eval_factory.contracts import (
    TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2,
    PromptLeakageCategoryV2,
    TaskDraftPromptSafetyStatusV2,
    TaskDraftV2,
    TaskPromptSafetyCheckOutcomeV2,
    TaskPromptSafetyGateStatusV2,
    TaskRequirementLineageV2,
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
    ContentRiskLabel,
    Disposition,
    OriginClass,
    ProvenanceDecision,
    TaintLabel,
    Visibility,
)
from eval_factory.contracts.task import (
    AttachmentCriticality,
    AttachmentDependency,
    EvidencePriority,
    RequirementConflict,
)
from eval_factory.task_authoring import (
    FakeTaskPromptSafetyFixture,
    FakeTaskPromptSafetyRunner,
    PromptLeakageReferenceSetCompiler,
    RestrictedPromptLeakageSource,
    TaskPromptSafetyCompiler,
    TaskPromptSafetyFindingFixture,
    TaskPromptSafetyOutcome,
    TaskPromptSafetyPolicyError,
    TaskPromptSafetyReason,
    TaskPromptSafetyRequest,
    TaskPromptSafetyRequestBuilder,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64


def _audit(
    created_at: datetime = datetime(2026, 7, 26, tzinfo=UTC),
    *,
    input_refs: tuple[ObjectRef, ...] = (),
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="task-prompt-safety-test",
        governing_versions=(VersionBinding(component="task-prompt-safety", version="r4-04-v1"),),
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


def _evidence(suffix: str = "intent") -> EvidenceRef:
    return EvidenceRef(
        evidence_ref_id=f"evidence-ref://prompt-safety/{suffix}",
        subject_ref=_ref("file-version-projection", suffix),
        source_spans=(
            SourceSpanRef(
                span_id=f"source-span://prompt-safety/{suffix}",
                source_trace_id="source-trace://prompt-safety",
                raw_sha256=HASH,
            ),
        ),
        polarity=EvidencePolarity.POSITIVE,
        capability="task-prompt-safety",
        capability_complete=True,
    )


def _lineage() -> TaskRequirementLineageV2:
    return TaskRequirementLineageV2(
        requirement_id="requirement://prompt-safety/primary",
        statement="Inspect the workspace and explain the observed design state.",
        criticality=AttachmentCriticality.CRITICAL,
        evidence_priority=EvidencePriority.DIRECT_OBSERVATION,
        evidence=(_evidence(),),
        task_episode_refs=(_ref("task-episode", "primary", version="v2"),),
        conflict_status=RequirementConflict.NONE,
    )


def _dependency() -> AttachmentDependency:
    return AttachmentDependency(
        dependency_id="attachment-dependency://prompt-safety/workspace",
        description="Use the provided input-state workspace.",
        criticality=AttachmentCriticality.REQUIRED,
        evidence_priority=EvidencePriority.EXPLICIT_REQUIREMENT,
        evidence=(_evidence("attachment"),),
    )


def _pending_draft(
    *,
    visible_prompt: str = "Inspect the workspace and explain the observed design state.",
    audit: ContractAudit | None = None,
) -> TaskDraftV2:
    draft = TaskDraftV2(
        task_draft_id="task-draft://sha256/" + HASH,
        task_version=1,
        supersedes_task_draft_ref=None,
        selection_context_ref=_ref("selection-context", "prompt-safety", version="v2"),
        task_episode_refs=(_ref("task-episode", "primary", version="v2"),),
        visible_prompt=visible_prompt,
        task_intent="Evaluate evidence-grounded workspace analysis.",
        evaluation_claim="The contestant must infer the design state from safe input evidence.",
        required_capabilities=("workspace-analysis",),
        allowed_tools=("file-read",),
        forbidden_outputs=(
            "Do not disclose the original final answer.",
            "Do not pre-place the completed deliverable.",
        ),
        attachment_dependencies=(_dependency(),),
        requirement_lineage=(_lineage(),),
        prompt_requirement_ids=("requirement://prompt-safety/primary",),
        uncertainties=(),
        prompt_safety_status=TaskDraftPromptSafetyStatusV2.PENDING,
        prompt_safety_gate_ref=None,
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


def _source(
    category: PromptLeakageCategoryV2,
    *,
    text: str | None = None,
    paths: tuple[str, ...] = (),
    suffix: str | None = None,
    capability_complete: bool = True,
) -> RestrictedPromptLeakageSource:
    source_suffix = suffix or category.value.casefold().replace("_", "-")
    material = text if text is not None else "\n".join(paths)
    digest = hashlib.sha256(material.encode()).hexdigest()
    object_type = {
        PromptLeakageCategoryV2.FINAL_ANSWER: "final-output",
        PromptLeakageCategoryV2.COMPLETED_DELIVERABLE: "completed-deliverable",
        PromptLeakageCategoryV2.PRIVATE_REFERENCE: "private-reference",
        PromptLeakageCategoryV2.GRADER_RULE: "grader-rule",
        PromptLeakageCategoryV2.HIDDEN_PASS_CONDITION: "hidden-pass-condition",
        PromptLeakageCategoryV2.HIDDEN_SELECTION_SIGNAL: "hidden-selection-signal",
        PromptLeakageCategoryV2.TRAJECTORY_SPECIFIC_STEP: "trajectory-specific-step",
    }[category]
    subject_ref = _ref(object_type, source_suffix, digest=digest)
    taints: frozenset[TaintLabel] = frozenset()
    risks: frozenset[ContentRiskLabel] = frozenset()
    origin = OriginClass.AGENT_GENERATED_INTERMEDIATE
    disposition = Disposition.QUARANTINE
    if category in {
        PromptLeakageCategoryV2.FINAL_ANSWER,
        PromptLeakageCategoryV2.COMPLETED_DELIVERABLE,
    }:
        origin = OriginClass.AGENT_GENERATED_FINAL
        taints = frozenset({TaintLabel.FINAL_OUTPUT_DERIVED})
        risks = frozenset({ContentRiskLabel.ANSWER_BEARING})
        disposition = Disposition.REJECT
    elif category is PromptLeakageCategoryV2.PRIVATE_REFERENCE:
        taints = frozenset({TaintLabel.PRIVATE_REFERENCE_DERIVED})
        risks = frozenset({ContentRiskLabel.ANSWER_BEARING})
        disposition = Disposition.REJECT
    elif category is PromptLeakageCategoryV2.GRADER_RULE:
        taints = frozenset({TaintLabel.GRADER_RULE_DERIVED})
        risks = frozenset({ContentRiskLabel.HIDDEN_PASS_CONDITION})
        disposition = Disposition.REJECT
    elif category is PromptLeakageCategoryV2.HIDDEN_PASS_CONDITION:
        risks = frozenset({ContentRiskLabel.HIDDEN_PASS_CONDITION})
        disposition = Disposition.REJECT
    decision = ProvenanceDecision(
        provenance_decision_id=f"provenance-decision://prompt-safety/{source_suffix}",
        subject_ref=subject_ref,
        origin_class=origin,
        taint_labels=taints,
        content_risk_labels=risks,
        visibility=Visibility.PRIVATE_STORE,
        disposition=disposition,
        rule_ids=("prompt-safety-source/test-v1",),
        source_event_refs=(_ref("trace-event", source_suffix),),
        confidence=1.0,
        review_required=True,
        policy_version="prompt-safety-source/test-v1",
        subject_sha256=digest,
        audit=_audit(),
    )
    return RestrictedPromptLeakageSource(
        source_id=f"restricted-prompt-leakage-source://{source_suffix}",
        category=category,
        source_subject_ref=subject_ref,
        source_provenance_decision=decision,
        classification_evidence_ref=_ref("safety-classification", source_suffix),
        source_text=text,
        logical_paths=paths,
        content_sha256=digest,
        capability_complete=capability_complete,
    )


def _reference_set(
    *sources: RestrictedPromptLeakageSource,
    complete_categories: frozenset[PromptLeakageCategoryV2] = (TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2),
    audit: ContractAudit | None = None,
):
    return PromptLeakageReferenceSetCompiler().compile(
        trace_envelope_ref=_ref("trace-envelope", "prompt-safety"),
        sources=tuple(sources),
        complete_categories=complete_categories,
        audit=audit or _audit(),
    )


def _request(
    draft: TaskDraftV2,
    reference_set,
    *,
    audit: ContractAudit | None = None,
):
    return TaskPromptSafetyRequestBuilder().build(
        task_draft=draft,
        leakage_reference_set=reference_set,
        model_profile="internal-task-prompt-safety-v1",
        prompt_version="task-prompt-safety-prompt/v1",
        audit=audit or _audit(),
    )


def _fixture(
    outcome: TaskPromptSafetyOutcome,
    *,
    findings: tuple[TaskPromptSafetyFindingFixture, ...] = (),
    reasons: frozenset[TaskPromptSafetyReason] = frozenset(),
    model_available: bool = True,
) -> FakeTaskPromptSafetyFixture:
    return FakeTaskPromptSafetyFixture(
        fixture_id=f"fake-task-prompt-safety-fixture://{outcome.value.casefold()}",
        outcome=outcome,
        findings=findings,
        unresolved_reasons=reasons,
        model_available=model_available,
    )


def _compile(
    draft: TaskDraftV2,
    reference_set,
    fixture: FakeTaskPromptSafetyFixture | None,
):
    request = _request(draft, reference_set)
    proposal = (
        None
        if fixture is None
        else FakeTaskPromptSafetyRunner().run(request, fixture=fixture, audit=_audit())
    )
    result = TaskPromptSafetyCompiler().compile(
        request=request,
        proposal=proposal,
        task_draft=draft,
        leakage_reference_set=reference_set,
        audit=_audit(),
    )
    return request, proposal, result


def test_clean_prompt_compiles_passed_gate_and_immutable_successor() -> None:
    draft = _pending_draft()
    reference_set = _reference_set()

    request, proposal, result = _compile(
        draft,
        reference_set,
        _fixture(TaskPromptSafetyOutcome.PASSED),
    )

    assert proposal is not None
    assert result.outcome is TaskPromptSafetyOutcome.PASSED
    assert result.task_prompt_safety_gate is not None
    assert result.task_prompt_safety_gate.status is TaskPromptSafetyGateStatusV2.PASSED
    assert all(
        item.outcome is TaskPromptSafetyCheckOutcomeV2.PASSED
        for item in result.task_prompt_safety_gate.checks
    )
    successor = result.task_draft
    assert successor is not None
    assert successor.task_version == draft.task_version + 1
    assert successor.supersedes_task_draft_ref == task_draft_ref(draft)
    assert successor.prompt_safety_status is TaskDraftPromptSafetyStatusV2.PASSED
    assert successor.prompt_safety_gate_ref is not None
    assert successor.visible_prompt == draft.visible_prompt
    assert successor.requirement_lineage == draft.requirement_lineage
    assert draft.prompt_safety_status is TaskDraftPromptSafetyStatusV2.PENDING
    assert request.untrusted_data_marker is True
    assert request.prompt_boundary_enforcement_ref.object_type == "prompt-boundary-enforcement"


def test_prompt_safety_request_hash_survives_json_round_trip() -> None:
    restricted_text = "The approved recommendation is reject the transaction"
    draft = _pending_draft(visible_prompt=f"Analyze this: {restricted_text}.")
    reference_set = _reference_set(
        _source(
            PromptLeakageCategoryV2.FINAL_ANSWER,
            text=restricted_text,
        )
    )
    request = _request(draft, reference_set)
    assert request.deterministic_scan.findings
    rendering = TaskPromptSafetyAgentInputV1(request=request)
    restored = TaskPromptSafetyAgentInputV1.model_validate_json(rendering.canonical_json()).request
    assert isinstance(restored, TaskPromptSafetyRequest)

    proposal = FakeTaskPromptSafetyRunner().run(
        restored,
        fixture=_fixture(TaskPromptSafetyOutcome.PASSED),
        audit=_audit(),
    )

    assert proposal.request_ref.object_sha256 == request.request_sha256


@pytest.mark.parametrize(
    ("category", "source_text", "paths", "visible_prompt"),
    [
        pytest.param(
            PromptLeakageCategoryV2.FINAL_ANSWER,
            "The approved recommendation is reject the transaction",
            (),
            "Analyze the evidence. The approved recommendation is reject the transaction.",
            id="verbatim-final-answer",
        ),
        pytest.param(
            PromptLeakageCategoryV2.FINAL_ANSWER,
            "The approved recommendation is reject the transaction",
            (),
            "Analyze evidence: THE approved   recommendation, is reject the transaction.",
            id="normalized-final-answer",
        ),
        pytest.param(
            PromptLeakageCategoryV2.FINAL_ANSWER,
            "The approved recommendation is reject the transaction because controls failed",
            (),
            "Use these facts: recommendation is reject the transaction because controls failed.",
            id="material-final-answer-fragment",
        ),
        pytest.param(
            PromptLeakageCategoryV2.COMPLETED_DELIVERABLE,
            None,
            ("deliverables/final_report.pdf",),
            "Open deliverables/final_report.pdf and submit it unchanged.",
            id="completed-output-path",
        ),
        pytest.param(
            PromptLeakageCategoryV2.PRIVATE_REFERENCE,
            "The private reference answer selects option delta",
            (),
            "Use the private reference answer selects option delta.",
            id="private-reference",
        ),
        pytest.param(
            PromptLeakageCategoryV2.GRADER_RULE,
            "Award full credit only when risk score exceeds ninety",
            (),
            "Ensure risk score exceeds ninety to receive full credit.",
            id="grader-rule",
        ),
        pytest.param(
            PromptLeakageCategoryV2.HIDDEN_PASS_CONDITION,
            "The hidden assertion requires exactly seven records",
            (),
            "Return exactly seven records because the hidden assertion requires it.",
            id="hidden-pass-condition",
        ),
        pytest.param(
            PromptLeakageCategoryV2.HIDDEN_SELECTION_SIGNAL,
            "The selected trace contains powershell error code forty two",
            (),
            "Reproduce powershell error code forty two from the selected trace.",
            id="hidden-selection-signal",
        ),
    ],
)
def test_known_restricted_fingerprints_block_without_semantic_override(
    category: PromptLeakageCategoryV2,
    source_text: str | None,
    paths: tuple[str, ...],
    visible_prompt: str,
) -> None:
    source = _source(category, text=source_text, paths=paths)
    reference_set = _reference_set(source)
    draft = _pending_draft(visible_prompt=visible_prompt)
    request = _request(draft, reference_set)

    assert request.deterministic_scan.findings
    _, _, result = _compile(draft, reference_set, None)

    assert result.outcome is TaskPromptSafetyOutcome.BLOCKED
    assert result.task_prompt_safety_gate is not None
    assert result.task_prompt_safety_gate.status is TaskPromptSafetyGateStatusV2.BLOCKED
    assert category in {item.category for item in result.task_prompt_safety_gate.findings}
    assert all(item.non_waivable for item in result.task_prompt_safety_gate.findings)
    successor = result.task_draft
    assert successor is not None
    assert successor.prompt_safety_status is TaskDraftPromptSafetyStatusV2.BLOCKED

    with pytest.raises(TaskPromptSafetyPolicyError, match="cannot clear deterministic"):
        _compile(
            draft,
            reference_set,
            _fixture(TaskPromptSafetyOutcome.PASSED),
        )


def test_known_finding_blocks_when_semantic_model_is_unavailable() -> None:
    source = _source(
        PromptLeakageCategoryV2.FINAL_ANSWER,
        text="The approved recommendation is reject the transaction",
    )
    reference_set = _reference_set(source)
    draft = _pending_draft(visible_prompt="The approved recommendation is reject the transaction.")

    _, _, result = _compile(
        draft,
        reference_set,
        _fixture(
            TaskPromptSafetyOutcome.BLOCKED_CAPABILITY,
            reasons=frozenset({TaskPromptSafetyReason.MODEL_UNAVAILABLE}),
            model_available=False,
        ),
    )

    assert result.outcome is TaskPromptSafetyOutcome.BLOCKED
    assert result.task_prompt_safety_gate is not None
    checks = {item.category: item.outcome for item in result.task_prompt_safety_gate.checks}
    assert checks[PromptLeakageCategoryV2.FINAL_ANSWER] is (TaskPromptSafetyCheckOutcomeV2.FAILED)
    assert checks[PromptLeakageCategoryV2.TRAJECTORY_SPECIFIC_STEP] is (
        TaskPromptSafetyCheckOutcomeV2.NOT_EVALUATED
    )


@pytest.mark.parametrize(
    "visible_prompt",
    [
        pytest.param(
            "Use the public specification to explain that the service is versioned.",
            id="safe-public-fact",
        ),
        pytest.param(
            "Analyze the user-provided conclusion and identify unsupported assumptions.",
            id="safe-user-conclusion",
        ),
    ],
)
def test_unrelated_restricted_sources_do_not_block_safe_prompt(
    visible_prompt: str,
) -> None:
    source = _source(
        PromptLeakageCategoryV2.FINAL_ANSWER,
        text="The approved recommendation is reject the transaction",
    )
    reference_set = _reference_set(source)
    draft = _pending_draft(visible_prompt=visible_prompt)

    request, _, result = _compile(
        draft,
        reference_set,
        _fixture(TaskPromptSafetyOutcome.PASSED),
    )

    assert request.deterministic_scan.findings == ()
    assert result.outcome is TaskPromptSafetyOutcome.PASSED


def test_semantic_accidental_path_and_copy_shortcut_block() -> None:
    draft = _pending_draft(
        visible_prompt=(
            "Run the exact original command, copy its generated file, rename it, and submit unchanged."
        )
    )
    reference_set = _reference_set()
    finding = TaskPromptSafetyFindingFixture(
        category=PromptLeakageCategoryV2.TRAJECTORY_SPECIFIC_STEP,
        prompt_span_start=0,
        prompt_span_end=len(draft.visible_prompt),
        rule_id="task-prompt-safety-semantic/trajectory-shortcut",
    )

    _, _, result = _compile(
        draft,
        reference_set,
        _fixture(TaskPromptSafetyOutcome.BLOCKED, findings=(finding,)),
    )

    assert result.outcome is TaskPromptSafetyOutcome.BLOCKED
    assert result.task_prompt_safety_gate is not None
    assert result.task_prompt_safety_gate.findings[0].category is (
        PromptLeakageCategoryV2.TRAJECTORY_SPECIFIC_STEP
    )


def test_incomplete_ambiguous_and_unavailable_paths_emit_no_artifacts() -> None:
    draft = _pending_draft()
    incomplete = _reference_set(
        complete_categories=frozenset(
            TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2 - {PromptLeakageCategoryV2.PRIVATE_REFERENCE}
        )
    )
    _, _, incomplete_result = _compile(
        draft,
        incomplete,
        _fixture(
            TaskPromptSafetyOutcome.BLOCKED_CAPABILITY,
            reasons=frozenset({TaskPromptSafetyReason.INCOMPLETE_REFERENCE_COVERAGE}),
        ),
    )
    assert incomplete_result.outcome is TaskPromptSafetyOutcome.BLOCKED_CAPABILITY
    assert incomplete_result.task_prompt_safety_gate is None
    assert incomplete_result.task_draft is None

    complete = _reference_set()
    _, _, abstain_result = _compile(
        draft,
        complete,
        _fixture(
            TaskPromptSafetyOutcome.ABSTAIN,
            reasons=frozenset({TaskPromptSafetyReason.AMBIGUOUS_TRAJECTORY}),
        ),
    )
    assert abstain_result.outcome is TaskPromptSafetyOutcome.ABSTAIN
    assert abstain_result.task_prompt_safety_gate is None
    assert abstain_result.task_draft is None

    _, _, unavailable_result = _compile(
        draft,
        complete,
        _fixture(
            TaskPromptSafetyOutcome.BLOCKED_CAPABILITY,
            reasons=frozenset({TaskPromptSafetyReason.MODEL_UNAVAILABLE}),
            model_available=False,
        ),
    )
    assert unavailable_result.outcome is TaskPromptSafetyOutcome.BLOCKED_CAPABILITY
    assert unavailable_result.task_prompt_safety_gate is None
    assert unavailable_result.task_draft is None


def test_semantic_request_and_outputs_do_not_disclose_restricted_material() -> None:
    secret_answer = "restricted final answer phrase never disclose"
    source = _source(PromptLeakageCategoryV2.FINAL_ANSWER, text=secret_answer)
    reference_set = _reference_set(source)
    draft = _pending_draft()
    request, proposal, result = _compile(
        draft,
        reference_set,
        _fixture(TaskPromptSafetyOutcome.PASSED),
    )

    for value in (request, proposal, result):
        rendered = str(value.model_dump(mode="json"))
        assert secret_answer not in rendered
        assert source.content_sha256 not in rendered
        assert source.source_subject_ref.object_id not in rendered
        assert source.source_provenance_decision.provenance_decision_id not in rendered
    assert request.leakage_reference_set_ref.object_type == "prompt-leakage-reference-set"
    assert set(request.audit.input_refs) == {
        request.deterministic_scan_ref,
        request.leakage_reference_set_ref,
        request.prompt_boundary_enforcement_ref,
        request.task_draft_ref,
    }


def test_reference_compiler_rejects_stale_sources_and_never_serializes_source_text() -> None:
    source = _source(
        PromptLeakageCategoryV2.FINAL_ANSWER,
        text="The final answer is a restricted recommendation",
    )
    reference_set = _reference_set(source)
    assert source.source_text not in str(reference_set.model_dump(mode="json"))

    stale = source.model_copy(update={"content_sha256": OTHER_HASH})
    with pytest.raises(TaskPromptSafetyPolicyError, match="content hash"):
        _reference_set(stale)
    with pytest.raises(TaskPromptSafetyPolicyError, match="duplicate"):
        _reference_set(source, source)
    with pytest.raises(ValidationError):
        RestrictedPromptLeakageSource.model_validate(
            {
                **source.model_dump(mode="json"),
                "raw_trace_text": "forbidden",
            }
        )


def test_reference_compiler_preserves_duplicate_content_lineage_and_normalizes_paths() -> None:
    first = _source(
        PromptLeakageCategoryV2.FINAL_ANSWER,
        text="The approved recommendation is reject the transaction",
        suffix="first",
    )
    second = _source(
        PromptLeakageCategoryV2.FINAL_ANSWER,
        text="The approved recommendation is reject the transaction",
        suffix="second",
    )
    reference_set = _reference_set(first, second)
    source_ids = {item.source_subject_ref.object_id for item in reference_set.fingerprints}
    assert source_ids == {
        first.source_subject_ref.object_id,
        second.source_subject_ref.object_id,
    }

    path_source = _source(
        PromptLeakageCategoryV2.COMPLETED_DELIVERABLE,
        paths=("deliverables/final_report.pdf",),
    ).model_copy(update={"logical_paths": ("deliverables\\final_report.pdf",)})
    path_set = _reference_set(path_source)
    path_request = _request(
        _pending_draft(visible_prompt="Open deliverables/final_report.pdf and submit it unchanged."),
        path_set,
    )
    assert path_request.deterministic_scan.findings


def test_compiler_rebuilds_and_rejects_draft_reference_request_and_proposal_tampering() -> None:
    draft = _pending_draft()
    reference_set = _reference_set()
    request = _request(draft, reference_set)
    fixture = _fixture(TaskPromptSafetyOutcome.PASSED)
    proposal = FakeTaskPromptSafetyRunner().run(request, fixture=fixture, audit=_audit())

    result = TaskPromptSafetyCompiler().compile(
        request=request,
        proposal=proposal,
        task_draft=draft,
        leakage_reference_set=reference_set,
        audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)),
    )
    assert result.outcome is TaskPromptSafetyOutcome.PASSED

    tampered_draft = draft.model_copy(update={"visible_prompt": draft.visible_prompt + " tampered"})
    with pytest.raises(TaskPromptSafetyPolicyError, match="TaskDraft"):
        TaskPromptSafetyCompiler().compile(
            request=request,
            proposal=proposal,
            task_draft=tampered_draft,
            leakage_reference_set=reference_set,
            audit=_audit(),
        )

    tampered_request = request.model_copy(update={"request_sha256": OTHER_HASH})
    with pytest.raises(TaskPromptSafetyPolicyError, match="request"):
        TaskPromptSafetyCompiler().compile(
            request=tampered_request,
            proposal=proposal,
            task_draft=draft,
            leakage_reference_set=reference_set,
            audit=_audit(),
        )

    tampered_proposal = proposal.model_copy(update={"proposal_sha256": OTHER_HASH})
    with pytest.raises(TaskPromptSafetyPolicyError, match="proposal"):
        TaskPromptSafetyCompiler().compile(
            request=request,
            proposal=tampered_proposal,
            task_draft=draft,
            leakage_reference_set=reference_set,
            audit=_audit(),
        )


def test_prompt_safety_ids_are_stable_across_audit_source_order_and_hash_seed(
    tmp_path: Path,
) -> None:
    first_source = _source(
        PromptLeakageCategoryV2.FINAL_ANSWER,
        text="The approved recommendation is reject the transaction",
        suffix="one",
    )
    second_source = _source(
        PromptLeakageCategoryV2.COMPLETED_DELIVERABLE,
        paths=("deliverables/final_report.pdf",),
        suffix="two",
    )
    first_set = _reference_set(
        first_source,
        second_source,
        audit=_audit(datetime(2026, 7, 26, tzinfo=UTC)),
    )
    second_set = _reference_set(
        second_source,
        first_source,
        audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)),
    )
    assert first_set.reference_set_id == second_set.reference_set_id
    assert first_set.reference_set_sha256 == second_set.reference_set_sha256
    assert first_set.canonical_sha256() != second_set.canonical_sha256()

    first_with_later_decision_audit = first_source.model_copy(
        update={
            "source_provenance_decision": first_source.source_provenance_decision.model_copy(
                update={"audit": _audit(datetime(2026, 7, 28, tzinfo=UTC))}
            )
        }
    )
    decision_audit_set = _reference_set(
        first_with_later_decision_audit,
        second_source,
    )
    assert decision_audit_set.reference_set_id == first_set.reference_set_id
    assert decision_audit_set.reference_set_sha256 == first_set.reference_set_sha256

    script = tmp_path / "check_task_prompt_safety_seed.py"
    script.write_text(
        """
from datetime import UTC, datetime
from eval_factory.contracts import PromptLeakageCategoryV2
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.safety import (
    ContentRiskLabel, Disposition, OriginClass, ProvenanceDecision, TaintLabel, Visibility,
)
from eval_factory.task_authoring import PromptLeakageReferenceSetCompiler, RestrictedPromptLeakageSource

digest = "66fc0f613cf7517d342ad2938f1f2064e78c9fd9d2affcd73e8c4ddbe2f969f6"
audit = ContractAudit(
    created_at=datetime(2026, 7, 26, tzinfo=UTC),
    created_by="seed-test",
    governing_versions=(VersionBinding(component="task-prompt-safety", version="r4-04-v1"),),
)
subject = ObjectRef(
    object_type="final-output",
    object_id="final-output://seed",
    object_version="v1",
    object_sha256=digest,
)
decision = ProvenanceDecision(
    provenance_decision_id="provenance-decision://seed",
    subject_ref=subject,
    origin_class=OriginClass.AGENT_GENERATED_FINAL,
    taint_labels=frozenset({TaintLabel.FINAL_OUTPUT_DERIVED}),
    content_risk_labels=frozenset({ContentRiskLabel.ANSWER_BEARING}),
    visibility=Visibility.PRIVATE_STORE,
    disposition=Disposition.REJECT,
    rule_ids=("seed-test/v1",),
    confidence=1.0,
    review_required=True,
    policy_version="seed-test/v1",
    subject_sha256=digest,
    audit=audit,
)
source = RestrictedPromptLeakageSource(
    source_id="restricted-prompt-leakage-source://seed",
    category=PromptLeakageCategoryV2.FINAL_ANSWER,
    source_subject_ref=subject,
    source_provenance_decision=decision,
    classification_evidence_ref=ObjectRef(
        object_type="safety-classification",
        object_id="safety-classification://seed",
        object_version="v1",
        object_sha256="a" * 64,
    ),
    source_text="The approved recommendation is reject the transaction",
    content_sha256=digest,
    capability_complete=True,
)
result = PromptLeakageReferenceSetCompiler().compile(
    trace_envelope_ref=ObjectRef(
        object_type="trace-envelope",
        object_id="trace-envelope://seed",
        object_version="v1",
        object_sha256="a" * 64,
    ),
    sources=(source,),
    complete_categories=frozenset(PromptLeakageCategoryV2),
    audit=audit,
)
print(result.reference_set_id)
print(result.reference_set_sha256)
""",
        encoding="utf-8",
    )
    outputs = []
    for seed in ("1", "99"):
        completed = subprocess.run(
            [sys.executable, str(script)],
            check=False,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src", "PYTHONHASHSEED": seed},
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(completed.stdout)
    assert len(set(outputs)) == 1
