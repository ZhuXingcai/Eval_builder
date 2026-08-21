from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts import (
    RubricJudgedObjectKindV2,
    RubricSourceModeV2,
    TaskDraftPromptSafetyStatusV2,
    TaskDraftV2,
    TaskRequirementLineageV2,
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
from eval_factory.contracts.task import (
    AttachmentCriticality,
    AttachmentDependency,
    EvidencePriority,
    RequirementConflict,
    RubricVisibility,
)
from eval_factory.task_authoring import (
    RUBRIC_AUTHORING_POLICY_VERSION,
    FakeRubricGenerationFixture,
    FakeRubricGenerationRunner,
    ImportedRubricDefinition,
    ImportedRubricStatus,
    RubricAuthoringOutcome,
    RubricAuthoringPolicyError,
    RubricAuthoringReason,
    RubricCandidateProposal,
    RubricCandidateRequest,
    RubricCandidateRequestBuilder,
    RubricCriterionSelection,
    RubricGeneratedSelectionAdapter,
    RubricImportAdapter,
    RubricSetCompiler,
    imported_rubric_definition_sha256,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64
THIRD_HASH = "c" * 64
PROMPT_REQUIREMENT_ID = "requirement://rubric/prompt"
SECOND_REQUIREMENT_ID = "requirement://rubric/second"
DEPENDENCY_ID = "attachment-dependency://rubric/workspace"
EVALUATOR_BINDING = "evaluator-binding://rubric/default"
ROOT = Path(__file__).resolve().parents[3]


def _audit(
    created_at: datetime = datetime(2026, 7, 26, tzinfo=UTC),
    *,
    input_refs: tuple[ObjectRef, ...] = (),
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="rubric-authoring-test",
        governing_versions=(VersionBinding(component="rubric-authoring", version="r4-05"),),
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


def _evidence(
    suffix: str,
    *,
    capability_complete: bool = True,
) -> EvidenceRef:
    return EvidenceRef(
        evidence_ref_id=f"evidence-ref://rubric/{suffix}",
        subject_ref=_ref("file-version-projection", suffix),
        source_spans=(
            SourceSpanRef(
                span_id=f"source-span://rubric/{suffix}",
                source_trace_id="source-trace://rubric-authoring",
                raw_sha256=HASH,
            ),
        ),
        polarity=EvidencePolarity.POSITIVE,
        capability="rubric-reachability",
        capability_complete=capability_complete,
    )


def _lineage(
    requirement_id: str,
    *,
    criticality: AttachmentCriticality,
    evidence: tuple[EvidenceRef, ...],
) -> TaskRequirementLineageV2:
    return TaskRequirementLineageV2(
        requirement_id=requirement_id,
        statement=f"Visible requirement for {requirement_id.rsplit('/', 1)[-1]}.",
        criticality=criticality,
        evidence_priority=EvidencePriority.EXPLICIT_REQUIREMENT,
        evidence=evidence,
        task_episode_refs=(_ref("task-episode", "rubric-primary", version="v2"),),
        conflict_status=RequirementConflict.NONE,
    )


def _dependency(
    *,
    evidence: tuple[EvidenceRef, ...] | None = None,
) -> AttachmentDependency:
    return AttachmentDependency(
        dependency_id=DEPENDENCY_ID,
        description="The input-state workspace used by the contestant.",
        criticality=AttachmentCriticality.REQUIRED,
        evidence_priority=EvidencePriority.EXPLICIT_REQUIREMENT,
        evidence=evidence or (_evidence("dependency"),),
    )


def _draft(
    *,
    audit: ContractAudit | None = None,
    prompt_safety_status: TaskDraftPromptSafetyStatusV2 = TaskDraftPromptSafetyStatusV2.PENDING,
    prompt_safety_gate_ref: ObjectRef | None = None,
    primary_complete: bool = True,
    attachment_dependencies: tuple[AttachmentDependency, ...] | None = None,
) -> TaskDraftV2:
    draft = TaskDraftV2(
        task_draft_id="task-draft://sha256/" + HASH,
        task_version=1,
        supersedes_task_draft_ref=None,
        selection_context_ref=_ref("selection-context", "rubric", version="v2"),
        task_episode_refs=(_ref("task-episode", "rubric-primary", version="v2"),),
        visible_prompt="Inspect the workspace and report the observable design state.",
        task_intent="Evaluate evidence-grounded workspace analysis.",
        evaluation_claim="The response must be derivable from visible requirements and input state.",
        required_capabilities=("workspace-analysis",),
        allowed_tools=("file-read",),
        forbidden_outputs=(
            "original final answer",
            "private grader controls",
        ),
        attachment_dependencies=(
            (_dependency(),) if attachment_dependencies is None else attachment_dependencies
        ),
        requirement_lineage=(
            _lineage(
                PROMPT_REQUIREMENT_ID,
                criticality=AttachmentCriticality.CRITICAL,
                evidence=(
                    _evidence(
                        "prompt-primary",
                        capability_complete=primary_complete,
                    ),
                ),
            ),
            _lineage(
                SECOND_REQUIREMENT_ID,
                criticality=AttachmentCriticality.OPTIONAL,
                evidence=(_evidence("prompt-second"),),
            ),
        ),
        prompt_requirement_ids=(PROMPT_REQUIREMENT_ID, SECOND_REQUIREMENT_ID),
        uncertainties=(),
        prompt_safety_status=prompt_safety_status,
        prompt_safety_gate_ref=prompt_safety_gate_ref,
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


def _selection(
    *,
    selection_id: str = "rubric-selection://response",
    kind: RubricJudgedObjectKindV2 = RubricJudgedObjectKindV2.CONTESTANT_RESPONSE,
    prompt_requirement_ids: tuple[str, ...] = (PROMPT_REQUIREMENT_ID,),
    attachment_dependency_ids: tuple[str, ...] = (),
    allowed_tool_ids: tuple[str, ...] = (),
    evaluator_binding: str = EVALUATOR_BINDING,
    weight: float = 1.0,
) -> RubricCriterionSelection:
    return RubricCriterionSelection(
        selection_id=selection_id,
        judged_object_id=f"judged-object://{selection_id.rsplit('/', 1)[-1]}",
        judged_object_kind=kind,
        judged_object_description="The exact object produced or changed by the contestant.",
        description="Satisfies the visible task requirement using reachable input-state evidence.",
        weight=weight,
        prompt_requirement_ids=prompt_requirement_ids,
        attachment_dependency_ids=attachment_dependency_ids,
        allowed_tool_ids=allowed_tool_ids,
        visibility=RubricVisibility.EVALUATOR_ONLY,
        evaluator_binding=evaluator_binding,
    )


def _request(
    draft: TaskDraftV2 | None = None,
    *,
    audit: ContractAudit | None = None,
) -> RubricCandidateRequest:
    return RubricCandidateRequestBuilder().build(
        task_draft=draft or _draft(),
        allowed_evaluator_bindings=(EVALUATOR_BINDING,),
        audit=audit or _audit(),
    )


def _generated_proposal(
    request: RubricCandidateRequest,
    *,
    criteria: tuple[RubricCriterionSelection, ...] | None = None,
    outcome: RubricAuthoringOutcome = RubricAuthoringOutcome.COMPILED,
    unresolved_reasons: frozenset[RubricAuthoringReason] = frozenset(),
    model_available: bool = True,
) -> RubricCandidateProposal:
    return FakeRubricGenerationRunner().run(
        request,
        fixture=FakeRubricGenerationFixture(
            fixture_id="fake-rubric-generation://primary",
            outcome=outcome,
            criteria=criteria or (() if outcome is not RubricAuthoringOutcome.COMPILED else (_selection(),)),
            unresolved_reasons=unresolved_reasons,
            model_available=model_available,
        ),
        model_profile="internal-rubric-author-v1",
        prompt_version="rubric-authoring-prompt/v1",
        audit=_audit(),
    )


def _imported(
    *,
    criteria: tuple[RubricCriterionSelection, ...] = (_selection(),),
    status: ImportedRubricStatus = ImportedRubricStatus.USABLE,
) -> ImportedRubricDefinition:
    imported = ImportedRubricDefinition(
        imported_rubric_ref=_ref("imported-rubric", "rubric-authoring-source"),
        status=status,
        criteria=criteria,
        definition_sha256="0" * 64,
        audit=_audit(input_refs=(_ref("private-grader-control", "must-not-cross"),)),
    )
    return imported.model_copy(update={"definition_sha256": imported_rubric_definition_sha256(imported)})


def test_request_contains_only_bounded_candidate_task_contract_data() -> None:
    caller_private = _ref("private-grader-control", "caller-private")
    request = _request(_draft(), audit=_audit(input_refs=(caller_private,)))
    payload = request.model_dump(mode="json")
    serialized = str(payload)

    assert request.task_draft_ref == task_draft_ref(_draft())
    assert request.candidate_task_projection_ref.object_type == "task-draft-rubric-input"
    assert request.prompt_boundary_enforcement_ref.object_type == "prompt-boundary-enforcement"
    assert request.visible_prompt == _draft().visible_prompt
    assert request.prompt_requirements[0].requirement_id == PROMPT_REQUIREMENT_ID
    assert request.attachment_dependencies[0].dependency_id == DEPENDENCY_ID
    assert request.allowed_evaluator_bindings == (EVALUATOR_BINDING,)
    assert request.policy_version == RUBRIC_AUTHORING_POLICY_VERSION
    assert "evidence_ref_id" not in serialized
    assert "source_spans" not in serialized
    assert "task-episode" not in serialized
    assert "private-grader-control" not in serialized
    assert "prompt-leakage" not in serialized


def test_imported_rubric_compiles_candidate_with_exact_provenance_and_evidence() -> None:
    draft = _draft()
    request = _request(draft)
    imported = _imported(
        criteria=(
            _selection(
                kind=RubricJudgedObjectKindV2.WORKSPACE_STATE,
                attachment_dependency_ids=(DEPENDENCY_ID,),
            ),
        )
    )
    proposal = RubricImportAdapter().adapt(request, imported=imported, audit=_audit())

    result = RubricSetCompiler().compile(
        request=request,
        proposal=proposal,
        task_draft=draft,
        audit=_audit(),
    )

    assert result.outcome is RubricAuthoringOutcome.COMPILED
    assert result.rubric_set is not None
    assert result.rubric_set.source_mode is RubricSourceModeV2.IMPORTED
    assert result.rubric_set.imported_from_ref == imported.imported_rubric_ref
    assert result.rubric_set.model_profile is None
    assert result.rubric_set.prompt_version is None
    assert result.rubric_set.criteria[0].approval_status == "CANDIDATE"
    assert result.rubric_set.criteria[0].reachability.evidence == (
        _evidence("dependency"),
        _evidence("prompt-primary"),
    )
    assert "private-grader-control://must-not-cross" not in str(result.model_dump(mode="json"))


def test_generated_rubric_compiles_model_metadata_and_candidate_only_approval() -> None:
    draft = _draft()
    request = _request(draft)
    with pytest.raises(RubricAuthoringPolicyError, match="imported rubric definition"):
        RubricImportAdapter().adapt(
            request,
            imported=_imported().model_copy(update={"definition_sha256": THIRD_HASH}),
            audit=_audit(),
        )
    proposal = _generated_proposal(request)

    result = RubricSetCompiler().compile(
        request=request,
        proposal=proposal,
        task_draft=draft,
        audit=_audit(),
    )

    assert result.outcome is RubricAuthoringOutcome.COMPILED
    assert result.rubric_set is not None
    assert result.rubric_set.source_mode is RubricSourceModeV2.GENERATED
    assert result.rubric_set.imported_from_ref is None
    assert result.rubric_set.model_profile == "internal-rubric-author-v1"
    assert result.rubric_set.prompt_version == "rubric-authoring-prompt/v1"
    assert result.rubric_set.task_draft_ref == task_draft_ref(draft)
    assert result.rubric_set.criteria[0].approval_status == "CANDIDATE"
    criterion = result.rubric_set.criteria[0]
    assert criterion.reachability.reachability_sha256 == rubric_reachability_carried_sha256(
        criterion.reachability
    )
    assert criterion.criterion_sha256 == rubric_criterion_carried_sha256(criterion)
    assert result.rubric_set.rubric_set_sha256 == rubric_set_carried_sha256(result.rubric_set)


def test_generated_selection_adapter_reuses_r4_proposal_authority() -> None:
    request = _request()
    criteria = (_selection(),)

    adapted = RubricGeneratedSelectionAdapter().adapt(
        request,
        outcome=RubricAuthoringOutcome.COMPILED,
        criteria=criteria,
        unresolved_reasons=frozenset(),
        model_profile="internal-rubric-author-v1",
        prompt_version="rubric-authoring-prompt/v1",
        audit=_audit(),
    )
    fake = _generated_proposal(
        request,
        criteria=criteria,
    )

    assert adapted == fake
    with pytest.raises(
        RubricAuthoringPolicyError,
        match="evaluator binding",
    ):
        RubricGeneratedSelectionAdapter().adapt(
            request,
            outcome=RubricAuthoringOutcome.COMPILED,
            criteria=(_selection(evaluator_binding="evaluator-binding://not-allowed"),),
            unresolved_reasons=frozenset(),
            model_profile="internal-rubric-author-v1",
            prompt_version="rubric-authoring-prompt/v1",
            audit=_audit(),
        )


def test_all_judged_object_kinds_compile_with_kind_specific_reachability() -> None:
    draft = _draft()
    request = _request(draft)
    criteria = (
        _selection(selection_id="rubric-selection://response"),
        _selection(
            selection_id="rubric-selection://artifact",
            kind=RubricJudgedObjectKindV2.OUTPUT_ARTIFACT,
        ),
        _selection(
            selection_id="rubric-selection://workspace",
            kind=RubricJudgedObjectKindV2.WORKSPACE_STATE,
            attachment_dependency_ids=(DEPENDENCY_ID,),
        ),
        _selection(
            selection_id="rubric-selection://tool",
            kind=RubricJudgedObjectKindV2.TOOL_BEHAVIOR,
            allowed_tool_ids=("file-read",),
        ),
        _selection(
            selection_id="rubric-selection://value",
            kind=RubricJudgedObjectKindV2.STRUCTURED_VALUE,
        ),
    )
    proposal = _generated_proposal(request, criteria=criteria)

    result = RubricSetCompiler().compile(
        request=request,
        proposal=proposal,
        task_draft=draft,
        audit=_audit(),
    )

    assert result.rubric_set is not None
    assert {item.judged_object.kind for item in result.rubric_set.criteria} == set(RubricJudgedObjectKindV2)
    workspace = next(
        item
        for item in result.rubric_set.criteria
        if item.judged_object.kind is RubricJudgedObjectKindV2.WORKSPACE_STATE
    )
    tool = next(
        item
        for item in result.rubric_set.criteria
        if item.judged_object.kind is RubricJudgedObjectKindV2.TOOL_BEHAVIOR
    )
    assert workspace.reachability.attachment_dependency_ids == (DEPENDENCY_ID,)
    assert tool.reachability.allowed_tool_ids == ("file-read",)


@pytest.mark.parametrize(
    ("draft", "selection", "reason"),
    [
        pytest.param(
            _draft(primary_complete=False),
            _selection(),
            RubricAuthoringReason.INCOMPLETE_REACHABILITY_EVIDENCE,
            id="incomplete-evidence",
        ),
        pytest.param(
            _draft(),
            _selection(
                kind=RubricJudgedObjectKindV2.WORKSPACE_STATE,
            ),
            RubricAuthoringReason.MISSING_ATTACHMENT_DEPENDENCY,
            id="workspace-without-dependency",
        ),
        pytest.param(
            _draft(),
            _selection(
                kind=RubricJudgedObjectKindV2.TOOL_BEHAVIOR,
            ),
            RubricAuthoringReason.MISSING_ALLOWED_TOOL,
            id="tool-behavior-without-tool",
        ),
    ],
)
def test_reachability_gaps_emit_blocked_without_partial_rubric(
    draft: TaskDraftV2,
    selection: RubricCriterionSelection,
    reason: RubricAuthoringReason,
) -> None:
    request = _request(draft)
    proposal = _generated_proposal(request, criteria=(selection,))

    result = RubricSetCompiler().compile(
        request=request,
        proposal=proposal,
        task_draft=draft,
        audit=_audit(),
    )

    assert result.outcome is RubricAuthoringOutcome.BLOCKED_REACHABILITY
    assert result.rubric_set is None
    assert reason in result.unresolved_reasons


@pytest.mark.parametrize(
    ("outcome", "reason", "model_available"),
    [
        pytest.param(
            RubricAuthoringOutcome.ABSTAIN,
            RubricAuthoringReason.AMBIGUOUS_RUBRIC,
            True,
            id="abstain",
        ),
        pytest.param(
            RubricAuthoringOutcome.BLOCKED_CAPABILITY,
            RubricAuthoringReason.MODEL_UNAVAILABLE,
            False,
            id="model-unavailable",
        ),
    ],
)
def test_generated_no_output_outcomes_never_fabricate_rubric_set(
    outcome: RubricAuthoringOutcome,
    reason: RubricAuthoringReason,
    model_available: bool,
) -> None:
    draft = _draft()
    request = _request(draft)
    proposal = _generated_proposal(
        request,
        outcome=outcome,
        unresolved_reasons=frozenset({reason}),
        model_available=model_available,
    )

    result = RubricSetCompiler().compile(
        request=request,
        proposal=proposal,
        task_draft=draft,
        audit=_audit(),
    )

    assert result.outcome is outcome
    assert result.rubric_set is None
    assert result.unresolved_reasons == frozenset({reason})


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        pytest.param(
            ImportedRubricStatus.REJECTED,
            RubricAuthoringReason.IMPORT_REJECTED,
            id="rejected",
        ),
        pytest.param(
            ImportedRubricStatus.AMBIGUOUS,
            RubricAuthoringReason.AMBIGUOUS_RUBRIC,
            id="ambiguous",
        ),
    ],
)
def test_unusable_import_emits_abstain_without_generic_fallback(
    status: ImportedRubricStatus,
    reason: RubricAuthoringReason,
) -> None:
    draft = _draft()
    request = _request(draft)
    proposal = RubricImportAdapter().adapt(
        request,
        imported=_imported(status=status),
        audit=_audit(),
    )

    result = RubricSetCompiler().compile(
        request=request,
        proposal=proposal,
        task_draft=draft,
        audit=_audit(),
    )

    assert result.outcome is RubricAuthoringOutcome.ABSTAIN
    assert result.rubric_set is None
    assert reason in result.unresolved_reasons


def test_semantic_models_reject_evidence_refs_hashes_versions_and_approval_authority() -> None:
    selection = _selection()
    forbidden_fields: dict[str, object] = {
        "evidence": (_evidence("injected"),),
        "task_draft_ref": task_draft_ref(_draft()),
        "criterion_sha256": HASH,
        "rubric_version": 2,
        "approval_status": "APPROVED",
        "policy_version": "attacker-policy",
        "imported_from_ref": _ref("imported-rubric", "injected"),
        "audit": _audit(),
    }

    for field_name, value in forbidden_fields.items():
        with pytest.raises(ValidationError):
            RubricCriterionSelection.model_validate(
                {
                    **selection.model_dump(mode="python"),
                    field_name: value,
                }
            )


def test_unknown_ids_and_unauthorized_evaluator_binding_fail_closed() -> None:
    draft = _draft()
    request = _request(draft)

    with pytest.raises(RubricAuthoringPolicyError, match="prompt requirement"):
        _generated_proposal(
            request,
            criteria=(
                _selection(
                    prompt_requirement_ids=("requirement://rubric/unknown",),
                ),
            ),
        )
    with pytest.raises(RubricAuthoringPolicyError, match="evaluator binding"):
        _generated_proposal(
            request,
            criteria=(_selection(evaluator_binding="evaluator-binding://unauthorized"),),
        )


def test_blocked_task_draft_and_stale_request_proposal_or_draft_fail_closed() -> None:
    gate_ref = _ref("task-prompt-safety-gate", "blocked", version="v2")
    blocked = _draft(
        prompt_safety_status=TaskDraftPromptSafetyStatusV2.BLOCKED,
        prompt_safety_gate_ref=gate_ref,
    )
    with pytest.raises(RubricAuthoringPolicyError, match="BLOCKED"):
        _request(blocked)

    draft = _draft()
    request = _request(draft)
    proposal = _generated_proposal(request)
    stale_request = request.model_copy(update={"request_sha256": THIRD_HASH})
    with pytest.raises(RubricAuthoringPolicyError, match="request"):
        RubricSetCompiler().compile(
            request=stale_request,
            proposal=proposal,
            task_draft=draft,
            audit=_audit(),
        )

    stale_proposal = proposal.model_copy(update={"proposal_sha256": THIRD_HASH})
    with pytest.raises(RubricAuthoringPolicyError, match="proposal"):
        RubricSetCompiler().compile(
            request=request,
            proposal=stale_proposal,
            task_draft=draft,
            audit=_audit(),
        )

    stale_draft = draft.model_copy(update={"visible_prompt": draft.visible_prompt + " changed"})
    with pytest.raises(RubricAuthoringPolicyError, match="TaskDraft"):
        RubricSetCompiler().compile(
            request=request,
            proposal=proposal,
            task_draft=stale_draft,
            audit=_audit(),
        )


def test_rubric_authoring_is_stable_across_audit_and_input_order() -> None:
    first_draft = _draft(audit=_audit(datetime(2026, 7, 26, tzinfo=UTC)))
    second_draft = _draft(audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)))
    first_request = RubricCandidateRequestBuilder().build(
        task_draft=first_draft,
        allowed_evaluator_bindings=(EVALUATOR_BINDING, "evaluator-binding://rubric/secondary"),
        audit=_audit(datetime(2026, 7, 26, tzinfo=UTC)),
    )
    second_request = RubricCandidateRequestBuilder().build(
        task_draft=second_draft,
        allowed_evaluator_bindings=("evaluator-binding://rubric/secondary", EVALUATOR_BINDING),
        audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)),
    )
    first_criteria = (
        _selection(selection_id="rubric-selection://first", weight=2.0),
        _selection(
            selection_id="rubric-selection://second",
            prompt_requirement_ids=(SECOND_REQUIREMENT_ID,),
            weight=1.0,
        ),
    )
    second_criteria = tuple(reversed(first_criteria))
    first_proposal = _generated_proposal(first_request, criteria=first_criteria)
    second_proposal = _generated_proposal(second_request, criteria=second_criteria)

    first = RubricSetCompiler().compile(
        request=first_request,
        proposal=first_proposal,
        task_draft=first_draft,
        audit=_audit(datetime(2026, 7, 26, tzinfo=UTC)),
    )
    second = RubricSetCompiler().compile(
        request=second_request,
        proposal=second_proposal,
        task_draft=second_draft,
        audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)),
    )

    assert first.rubric_set is not None
    assert second.rubric_set is not None
    assert first_request.request_sha256 == second_request.request_sha256
    assert first.rubric_set.rubric_set_id == second.rubric_set.rubric_set_id
    assert first.rubric_set.rubric_set_sha256 == second.rubric_set.rubric_set_sha256


def test_task_safety_status_and_rubric_source_mode_have_distinct_identities() -> None:
    pending = _draft()
    passed = _draft(
        prompt_safety_status=TaskDraftPromptSafetyStatusV2.PASSED,
        prompt_safety_gate_ref=_ref(
            "task-prompt-safety-gate",
            "passed",
            version="v2",
        ),
    )
    pending_request = _request(pending)
    passed_request = _request(passed)

    assert pending_request.rubric_candidate_request_id != (passed_request.rubric_candidate_request_id)

    generated = RubricSetCompiler().compile(
        request=pending_request,
        proposal=_generated_proposal(pending_request),
        task_draft=pending,
        audit=_audit(),
    )
    imported = RubricSetCompiler().compile(
        request=pending_request,
        proposal=RubricImportAdapter().adapt(
            pending_request,
            imported=_imported(),
            audit=_audit(),
        ),
        task_draft=pending,
        audit=_audit(),
    )

    assert generated.rubric_set is not None
    assert imported.rubric_set is not None
    assert generated.rubric_set.rubric_set_id != imported.rubric_set.rubric_set_id


def test_rubric_authoring_is_stable_across_python_hash_seed() -> None:
    code = """
import runpy

ns = runpy.run_path("tests/eval_factory/unit/test_rubric_authoring.py")
draft = ns["_draft"]()
request = ns["_request"](draft)
proposal = ns["_generated_proposal"](request)
result = ns["RubricSetCompiler"]().compile(
    request=request,
    proposal=proposal,
    task_draft=draft,
    audit=ns["_audit"](),
)
assert result.rubric_set is not None
print(request.rubric_candidate_request_id)
print(proposal.rubric_candidate_proposal_id)
print(result.rubric_set.rubric_set_id)
print(result.rubric_authoring_result_id)
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
